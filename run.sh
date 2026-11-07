#!/usr/bin/env bash
# ==============================================================================
# Spring AI + Ollama + RAG + React Runner Script
# ==============================================================================

set -euo pipefail

# Resolve project root directory regardless of current working directory
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Color support for interactive terminals
if [ -t 1 ]; then
  GREEN='\033[0;32m'
  BLUE='\033[0;34m'
  YELLOW='\033[1;33m'
  RED='\033[0;31m'
  CYAN='\033[0;36m'
  BOLD='\033[1m'
  NC='\033[0m'
else
  GREEN=''
  BLUE=''
  YELLOW=''
  RED=''
  CYAN=''
  BOLD=''
  NC=''
fi

log_info()    { printf "${CYAN}ℹ️  %s${NC}\n" "$*"; }
log_success() { printf "${GREEN}✅ %s${NC}\n" "$*"; }
log_warn()    { printf "${YELLOW}⚠️  %s${NC}\n" "$*"; }
log_error()   { printf "${RED}❌ %s${NC}\n" "$*" >&2; }
log_step()    { printf "${BOLD}${BLUE}==> %s${NC}\n" "$*"; }

# ------------------------------------------------------------------------------
# Environment Variables & Configuration
# ------------------------------------------------------------------------------
load_env() {
  if [ -f "$ROOT_DIR/.env" ]; then
    log_info "Loading configuration from .env file..."
    set -a
    # shellcheck disable=SC1091
    source "$ROOT_DIR/.env"
    set +a
  elif [ -f "$ROOT_DIR/.env.example" ]; then
    log_info "No .env file found. Creating .env from .env.example..."
    cp "$ROOT_DIR/.env.example" "$ROOT_DIR/.env"
    set -a
    # shellcheck disable=SC1091
    source "$ROOT_DIR/.env"
    set +a
  fi

  # Default environment fallbacks
  export DB_HOST="${DB_HOST:-localhost}"
  export DB_PORT="${DB_PORT:-5432}"
  export DB_NAME="${DB_NAME:-ragdb}"
  export DB_USER="${DB_USER:-postgres}"
  export DB_PASSWORD="${DB_PASSWORD:-postgres}"
  export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
  export OLLAMA_CHAT_MODEL="${OLLAMA_CHAT_MODEL:-qwen2.5:1.5b}"
  export OLLAMA_EMBED_MODEL="${OLLAMA_EMBED_MODEL:-nomic-embed-text}"
  export VECTOR_DIMENSIONS="${VECTOR_DIMENSIONS:-768}"
}

# ------------------------------------------------------------------------------
# Java & Maven Environment Detection
# ------------------------------------------------------------------------------
detect_java() {
  if [ -z "${JAVA_HOME:-}" ]; then
    if [ -x "/usr/libexec/java_home" ]; then
      JAVA_HOME=$(/usr/libexec/java_home 2>/dev/null || true)
    fi
    if [ -z "${JAVA_HOME:-}" ] && [ -d "/opt/homebrew/opt/openjdk" ]; then
      JAVA_HOME="/opt/homebrew/opt/openjdk"
    fi
  fi

  if [ -n "${JAVA_HOME:-}" ]; then
    export JAVA_HOME
    export PATH="$JAVA_HOME/bin:$PATH"
  fi

  if ! command -v java >/dev/null 2>&1; then
    log_error "Java (JDK 21+) is not installed or not in PATH."
    log_info  "Install JDK 21+ (e.g. 'brew install openjdk' or download from adoptium.net)."
    exit 1
  fi
}

# ------------------------------------------------------------------------------
# Docker Compose Detection
# ------------------------------------------------------------------------------
detect_docker_compose() {
  if docker compose version >/dev/null 2>&1; then
    DOCKER_COMPOSE="docker compose"
  elif command -v docker-compose >/dev/null 2>&1; then
    DOCKER_COMPOSE="docker-compose"
  else
    log_error "Docker Compose is required but neither 'docker compose' nor 'docker-compose' was found."
    log_info  "Please install Docker Desktop or Docker Compose plugin."
    exit 1
  fi

  if ! docker info >/dev/null 2>&1; then
    log_warn "Docker daemon is not running or not accessible."
    log_info "Please ensure Docker Desktop / the Docker daemon is started."
  fi
}

# ------------------------------------------------------------------------------
# Preflight Requirements Check
# ------------------------------------------------------------------------------
check_requirements() {
  log_step "Checking preflight requirements..."
  load_env
  detect_docker_compose
  detect_java

  if ! command -v mvn >/dev/null 2>&1; then
    log_error "Maven 3.9+ is required but not found in PATH."
    log_info  "Install Maven (e.g. 'brew install maven')."
    exit 1
  fi

  if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1; then
    log_error "Node.js (v20+) and npm are required but not found in PATH."
    log_info  "Install Node.js (e.g. 'brew install node' or using nvm/fnm)."
    exit 1
  fi

  log_success "All prerequisite CLI tools are available."
}

# ------------------------------------------------------------------------------
# Wait Helpers
# ------------------------------------------------------------------------------
wait_for_postgres() {
  log_info "Waiting for PostgreSQL (PGVector) to accept connections on port ${DB_PORT}..."
  local retries=30
  local count=0
  until $DOCKER_COMPOSE exec -T postgres pg_isready -U "${DB_USER}" -d "${DB_NAME}" >/dev/null 2>&1; do
    count=$((count + 1))
    if [ "$count" -ge "$retries" ]; then
      log_error "Timed out waiting for PostgreSQL container to become ready."
      exit 1
    fi
    sleep 1
  done
  log_success "PostgreSQL is ready."
}

wait_for_ollama() {
  log_info "Waiting for Ollama service to be ready..."
  local retries=30
  local count=0
  until $DOCKER_COMPOSE exec -T ollama ollama list >/dev/null 2>&1; do
    count=$((count + 1))
    if [ "$count" -ge "$retries" ]; then
      log_error "Timed out waiting for Ollama container to become ready."
      exit 1
    fi
    sleep 2
  done
  log_success "Ollama service is ready."
}

# ------------------------------------------------------------------------------
# Ollama Model Pulling
# ------------------------------------------------------------------------------
pull_model_if_missing() {
  local model="$1"
  if [ -z "$model" ]; then
    return
  fi

  log_info "Checking Ollama model: ${model}..."
  if $DOCKER_COMPOSE exec -T ollama ollama list 2>/dev/null | grep -q "${model}"; then
    log_success "Model '${model}' is already available in Ollama."
  else
    log_info "Downloading model '${model}' (this may take several minutes on first download)..."
    $DOCKER_COMPOSE exec -T ollama ollama pull "$model"
    log_success "Model '${model}' downloaded successfully."
  fi
}

pull_models() {
  log_step "Ensuring required Ollama models are pulled..."
  load_env
  detect_docker_compose
  pull_model_if_missing "$OLLAMA_CHAT_MODEL"
  pull_model_if_missing "$OLLAMA_EMBED_MODEL"
}

# ------------------------------------------------------------------------------
# Infrastructure Management
# ------------------------------------------------------------------------------
start_infra() {
  log_step "Starting infrastructure (PostgreSQL/PGVector & Ollama)..."
  load_env
  detect_docker_compose

  $DOCKER_COMPOSE up -d
  wait_for_postgres
  wait_for_ollama
  pull_models
  log_success "Infrastructure is up and models are ready."
}

stop_infra() {
  log_step "Stopping Docker infrastructure..."
  detect_docker_compose
  $DOCKER_COMPOSE down
  log_success "Docker containers stopped."
}

# ------------------------------------------------------------------------------
# Backend & Frontend Run
# ------------------------------------------------------------------------------
run_backend() {
  log_step "Starting Spring Boot backend..."
  load_env
  detect_java
  cd "$ROOT_DIR/backend"
  local JAR_FILE="$ROOT_DIR/backend/target/spring-ai-rag-demo-0.0.1-SNAPSHOT.jar"
  if [ -f "$JAR_FILE" ]; then
    log_info "Running packaged jar (-Xmx512m)..."
    exec java -Xmx512m -jar "$JAR_FILE" "$@"
  else
    mvn spring-boot:run "$@"
  fi
}

setup_frontend() {
  if [ ! -d "$ROOT_DIR/frontend/node_modules" ]; then
    log_info "node_modules not found. Installing frontend dependencies..."
    (cd "$ROOT_DIR/frontend" && npm install)
    log_success "Frontend dependencies installed."
  fi
}

run_frontend() {
  log_step "Starting React frontend (Vite)..."
  setup_frontend
  cd "$ROOT_DIR/frontend"
  npm run dev -- "$@"
}

# ------------------------------------------------------------------------------
# Combined Run (All Services)
# ------------------------------------------------------------------------------
run_all() {
  check_requirements
  start_infra
  setup_frontend

  printf "\n"
  echo "=================================================================="
  printf "${BOLD}${GREEN}🚀 Spring AI + Ollama + RAG Demo is starting!${NC}\n"
  echo "=================================================================="
  printf "  • ${BOLD}React Frontend UI:${NC}    http://localhost:5173\n"
  printf "  • ${BOLD}Spring Boot API:${NC}      http://localhost:8080\n"
  printf "  • ${BOLD}Actuator Health:${NC}      http://localhost:8080/actuator/health\n"
  printf "  • ${BOLD}Ollama Inference:${NC}     http://localhost:11434\n"
  printf "  • ${BOLD}PostgreSQL / PGVector:${NC} localhost:5432 (DB: %s)\n" "$DB_NAME"
  printf "  • ${BOLD}Chat Model:${NC}           %s\n" "$OLLAMA_CHAT_MODEL"
  printf "  • ${BOLD}Embedding Model:${NC}      %s (dim: %s)\n" "$OLLAMA_EMBED_MODEL" "$VECTOR_DIMENSIONS"
  echo "=================================================================="
  printf "${YELLOW}Press Ctrl+C to shut down the backend and frontend.${NC}\n\n"

  BACKEND_PID=""
  FRONTEND_PID=""

  cleanup() {
    printf "\n"
    log_warn "Shutting down application processes..."
    if [ -n "${BACKEND_PID:-}" ] && kill -0 "$BACKEND_PID" 2>/dev/null; then
      kill "$BACKEND_PID" 2>/dev/null || true
    fi
    if [ -n "${FRONTEND_PID:-}" ] && kill -0 "$FRONTEND_PID" 2>/dev/null; then
      kill "$FRONTEND_PID" 2>/dev/null || true
    fi
    wait "${BACKEND_PID:-}" 2>/dev/null || true
    wait "${FRONTEND_PID:-}" 2>/dev/null || true
    log_success "Spring Boot and React frontend stopped."
    log_info "Note: Docker containers (PostgreSQL & Ollama) are still running in the background."
    log_info "To stop them, run: ./run.sh stop"
    exit 0
  }

  trap cleanup INT TERM

  log_step "Launching Spring Boot backend..."
  local JAR_FILE="$ROOT_DIR/backend/target/spring-ai-rag-demo-0.0.1-SNAPSHOT.jar"
  if [ -f "$JAR_FILE" ]; then
    (cd "$ROOT_DIR/backend" && exec java -Xmx512m -jar "$JAR_FILE") &
  else
    (cd "$ROOT_DIR/backend" && exec mvn spring-boot:run) &
  fi
  BACKEND_PID=$!

  log_step "Launching React frontend..."
  (cd "$ROOT_DIR/frontend" && exec npm run dev) &
  FRONTEND_PID=$!

  # Wait for child processes
  wait "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
}

# ------------------------------------------------------------------------------
# Status Check
# ------------------------------------------------------------------------------
show_status() {
  load_env
  detect_docker_compose

  echo "=================================================================="
  printf "${BOLD}Spring AI RAG Demo - Status Overview${NC}\n"
  echo "=================================================================="
  echo ""
  printf "${BOLD}🐳 Docker Containers:${NC}\n"
  $DOCKER_COMPOSE ps
  echo ""
  printf "${BOLD}🧠 Installed Ollama Models:${NC}\n"
  $DOCKER_COMPOSE exec -T ollama ollama list 2>/dev/null || echo "Ollama container is not currently running."
  echo ""
  printf "${BOLD}🌐 Service Health Checks:${NC}\n"
  if curl -sf http://localhost:8080/actuator/health >/dev/null 2>&1; then
    printf "  • Spring Boot (8080):      ${GREEN}✅ RUNNING (Health OK)${NC}\n"
  else
    printf "  • Spring Boot (8080):      ${RED}❌ NOT REACHABLE${NC}\n"
  fi

  if curl -sf http://localhost:5173 >/dev/null 2>&1; then
    printf "  • React Frontend (5173):   ${GREEN}✅ RUNNING${NC}\n"
  else
    printf "  • React Frontend (5173):   ${RED}❌ NOT REACHABLE${NC}\n"
  fi

  if curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
    printf "  • Ollama API (11434):      ${GREEN}✅ RUNNING${NC}\n"
  else
    printf "  • Ollama API (11434):      ${RED}❌ NOT REACHABLE${NC}\n"
  fi
  echo "=================================================================="
}

# ------------------------------------------------------------------------------
# Clean Artifacts
# ------------------------------------------------------------------------------
clean_project() {
  load_env
  detect_docker_compose
  detect_java

  log_step "Stopping containers..."
  $DOCKER_COMPOSE down

  log_step "Cleaning backend Maven build..."
  (cd "$ROOT_DIR/backend" && mvn clean)

  printf "Remove frontend/node_modules directory? [y/N]: "
  read -r response
  if [[ "$response" =~ ^[Yy]$ ]]; then
    rm -rf "$ROOT_DIR/frontend/node_modules"
    log_success "frontend/node_modules removed."
  fi

  log_success "Clean completed."
}

# ------------------------------------------------------------------------------
# Help
# ------------------------------------------------------------------------------
show_help() {
  cat << EOF
Spring AI + Ollama + RAG + React Runner

Usage:
  ./run.sh [command] [args...]

Commands:
  all (default)    Check requirements, start Docker infra, pull models, and run backend + frontend.
  infra            Start PostgreSQL (PGVector) and Ollama in Docker, and pull models.
  models           Pull Ollama chat and embedding models configured in .env.
  backend          Start only the Spring Boot backend (port 8080).
  frontend         Start only the React Vite frontend (port 5173).
  status           Inspect Docker containers, Ollama models, and service health endpoints.
  stop | down      Stop all Docker infrastructure containers.
  restart          Restart Docker infrastructure and full application.
  clean            Stop containers, clean Maven build artifacts, and optionally remove node_modules.
  help | -h        Display this help message.

Environment Variables (configured via .env or shell):
  OLLAMA_CHAT_MODEL     Chat model (default: qwen2.5:1.5b)
  OLLAMA_EMBED_MODEL    Embedding model (default: nomic-embed-text)
  OLLAMA_BASE_URL       Ollama base URL (default: http://localhost:11434)
  DB_HOST               PostgreSQL host (default: localhost)
  DB_PORT               PostgreSQL port (default: 5432)
  DB_NAME               PostgreSQL database (default: ragdb)
  DB_USER               PostgreSQL username (default: postgres)
  DB_PASSWORD           PostgreSQL password (default: postgres)
  VECTOR_DIMENSIONS     Vector dimension size (default: 768)

Examples:
  ./run.sh                  # Start the entire demo
  ./run.sh infra            # Start only PostgreSQL and Ollama
  ./run.sh backend          # Start only backend
  ./run.sh frontend         # Start only React UI
  ./run.sh stop             # Stop Docker containers
  ./run.sh status           # Check health of running services
EOF
}

# ------------------------------------------------------------------------------
# Command Dispatcher
# ------------------------------------------------------------------------------
COMMAND="${1:-all}"
shift || true

case "$COMMAND" in
  all|start)
    run_all "$@"
    ;;
  infra)
    start_infra "$@"
    ;;
  models|pull-models)
    pull_models "$@"
    ;;
  backend)
    run_backend "$@"
    ;;
  frontend)
    run_frontend "$@"
    ;;
  status)
    show_status "$@"
    ;;
  stop|down)
    stop_infra "$@"
    ;;
  restart)
    stop_infra
    run_all "$@"
    ;;
  clean)
    clean_project "$@"
    ;;
  help|-h|--help)
    show_help
    ;;
  *)
    log_error "Unknown command: '$COMMAND'"
    printf "\n"
    show_help
    exit 1
    ;;
esac

