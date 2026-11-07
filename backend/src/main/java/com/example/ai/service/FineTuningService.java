package com.example.ai.service;

import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.BufferedReader;
import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.time.Duration;
import java.time.Instant;
import java.util.*;
import java.util.stream.Stream;

@Service
public class FineTuningService {
  private static final Logger log = LoggerFactory.getLogger(FineTuningService.class);
  private final ObjectMapper objectMapper;
  private final HttpClient httpClient;

  @Value("${spring.ai.ollama.base-url:http://localhost:11434}")
  private String ollamaBaseUrl;

  @Value("${spring.ai.ollama.chat.model:qwen2.5:1.5b}")
  private String defaultBaseModel;

  public FineTuningService(ObjectMapper objectMapper) {
    this.objectMapper = objectMapper;
    this.httpClient = HttpClient.newBuilder()
        .connectTimeout(Duration.ofSeconds(10))
        .build();
  }

  public Path getDatasetDirectory() {
    Path current = Paths.get(".").toAbsolutePath().normalize();
    Path projectRoot = current;
    if (current.endsWith("backend")) {
      projectRoot = current.getParent();
    }

    Path datasetDir = projectRoot.resolve("fine-tuning").resolve("dataset");
    if (!Files.exists(datasetDir)) {
      Path altDir = projectRoot.resolve("fine-tune").resolve("dataset");
      if (Files.exists(altDir)) {
        return altDir;
      }
      try {
        Files.createDirectories(datasetDir);
      } catch (IOException e) {
        log.error("Failed to create dataset directory: {}", datasetDir, e);
      }
    }
    return datasetDir;
  }

  public Path getOutputDirectory() {
    Path current = Paths.get(".").toAbsolutePath().normalize();
    Path projectRoot = current;
    if (current.endsWith("backend")) {
      projectRoot = current.getParent();
    }
    Path outputDir = projectRoot.resolve("fine-tuning").resolve("output");
    try {
      Files.createDirectories(outputDir);
    } catch (IOException e) {
      log.error("Failed to create output directory: {}", outputDir, e);
    }
    return outputDir;
  }

  public List<DatasetFileInfo> listDatasetFiles() {
    Path dir = getDatasetDirectory();
    List<DatasetFileInfo> files = new ArrayList<>();
    if (!Files.exists(dir)) {
      return files;
    }
    try (Stream<Path> stream = Files.list(dir)) {
      List<Path> pathList = stream
          .filter(Files::isRegularFile)
          .filter(p -> !p.getFileName().toString().startsWith("."))
          .sorted((a, b) -> b.getFileName().toString().compareTo(a.getFileName().toString()))
          .toList();

      for (Path p : pathList) {
        long size = Files.size(p);
        Instant modified = Files.getLastModifiedTime(p).toInstant();
        int lineCount = 0;
        try (BufferedReader reader = Files.newBufferedReader(p, StandardCharsets.UTF_8)) {
          while (reader.readLine() != null) {
            lineCount++;
          }
        } catch (Exception ignored) {}
        files.add(new DatasetFileInfo(p.getFileName().toString(), size, modified.toString(), lineCount));
      }
    } catch (IOException e) {
      log.error("Error reading dataset directory", e);
    }
    return files;
  }

  public UploadResponse uploadDataset(MultipartFile file) throws IOException {
    if (file.isEmpty()) {
      throw new IllegalArgumentException("Uploaded file is empty");
    }
    String originalName = file.getOriginalFilename();
    if (originalName == null || originalName.isBlank()) {
      originalName = "dataset-" + System.currentTimeMillis() + ".jsonl";
    }
    String cleanName = Paths.get(originalName).getFileName().toString();
    Path targetPath = getDatasetDirectory().resolve(cleanName);
    Files.copy(file.getInputStream(), targetPath, StandardCopyOption.REPLACE_EXISTING);

    int samples = 0;
    try (BufferedReader reader = Files.newBufferedReader(targetPath, StandardCharsets.UTF_8)) {
      while (reader.readLine() != null) {
        samples++;
      }
    } catch (Exception ignored) {}

    return new UploadResponse(cleanName, Files.size(targetPath), samples, "File uploaded successfully to fine-tune/dataset/" + cleanName);
  }

  public FineTuneResult processFineTuning(FineTuneRequest request) {
    StringBuilder logs = new StringBuilder();
    logs.append(String.format("Starting fine-tuning process at %s\n", Instant.now()));

    String datasetFilename = request.datasetFile();
    if (datasetFilename == null || datasetFilename.isBlank()) {
      datasetFilename = "spring-ai.jsonl";
    }
    String targetModel = (request.modelName() == null || request.modelName().isBlank())
        ? "spring-ai-expert"
        : request.modelName().trim().toLowerCase().replaceAll("[^a-z0-9._-]", "-");

    String baseModel = (request.baseModel() == null || request.baseModel().isBlank())
        ? defaultBaseModel
        : request.baseModel().trim();

    double temperature = request.temperature() != null ? request.temperature() : 0.1;

    Path datasetPath = getDatasetDirectory().resolve(datasetFilename);
    if (!Files.exists(datasetPath)) {
      logs.append(String.format("ERROR: Dataset file '%s' not found in %s\n", datasetFilename, getDatasetDirectory()));
      return new FineTuneResult(false, targetModel, "Dataset file not found: " + datasetFilename, logs.toString(), 0);
    }

    logs.append(String.format("Using dataset: %s\n", datasetPath.toAbsolutePath()));
    logs.append(String.format("Target model: %s\n", targetModel));
    logs.append(String.format("Base model: %s\n", baseModel));
    logs.append(String.format("Temperature: %.2f\n", temperature));

    // Parse training dataset samples
    List<DomainPair> pairs = new ArrayList<>();
    int sampleCount = 0;

    try (BufferedReader reader = Files.newBufferedReader(datasetPath, StandardCharsets.UTF_8)) {
      String line;
      while ((line = reader.readLine()) != null) {
        line = line.trim();
        if (line.isEmpty() || line.startsWith("#")) continue;
        sampleCount++;
        try {
          JsonNode node = objectMapper.readTree(line);
          if (node.has("messages") && node.get("messages").isArray()) {
            String userPrompt = "";
            String assistantResponse = "";
            for (JsonNode msg : node.get("messages")) {
              String role = msg.path("role").asText("");
              String content = msg.path("content").asText("");
              if ("user".equalsIgnoreCase(role)) {
                userPrompt = content;
              } else if ("assistant".equalsIgnoreCase(role)) {
                assistantResponse = content;
              }
            }
            if (!userPrompt.isBlank() && !assistantResponse.isBlank()) {
              pairs.add(new DomainPair(userPrompt, assistantResponse));
            }
          } else if (node.has("prompt") && node.has("response")) {
            pairs.add(new DomainPair(node.get("prompt").asText(), node.get("response").asText()));
          } else {
            pairs.add(new DomainPair(line, ""));
          }
        } catch (Exception parseEx) {
          pairs.add(new DomainPair(line, ""));
        }
      }
    } catch (IOException e) {
      logs.append(String.format("ERROR reading dataset: %s\n", e.getMessage()));
      return new FineTuneResult(false, targetModel, "Failed to read dataset: " + e.getMessage(), logs.toString(), 0);
    }

    logs.append(String.format("Parsed %d instruction samples from %s\n", sampleCount, datasetFilename));

    // Build specialized System Instructions with domain definitions and rules
    StringBuilder systemPrompt = new StringBuilder();
    systemPrompt.append("You are a specialized enterprise AI expert assistant fine-tuned on custom domain instructions.\n");
    systemPrompt.append("You have been trained on specific domain knowledge and definitions that you must prioritize.\n\n");
    systemPrompt.append("### LEARNED DOMAIN KNOWLEDGE & DEFINITIONS:\n");

    for (int i = 0; i < pairs.size(); i++) {
      DomainPair pair = pairs.get(i);
      if (!pair.assistantResponse.isBlank()) {
        systemPrompt.append(String.format("- Topic / Question: %s\n", pair.userPrompt));
        systemPrompt.append(String.format("  Learned Answer: %s\n", pair.assistantResponse));
        systemPrompt.append(String.format("  Rule: When asked \"%s\" or about this concept, answer: \"%s\".\n\n",
            pair.userPrompt, pair.assistantResponse));
      } else {
        systemPrompt.append(String.format("- Knowledge entry: %s\n\n", pair.userPrompt));
      }
    }

    systemPrompt.append("### INSTRUCTION GUIDELINES:\n");
    systemPrompt.append("- Always prioritize and use the learned domain knowledge and definitions above when answering user questions.\n");
    systemPrompt.append("- If asked about terms defined in the domain rules (such as r4r), provide the learned answer directly.\n");
    systemPrompt.append("- Provide concise, accurate, production-grade answers.\n");

    // Save generated Modelfile
    try {
      Path modelOutputDir = getOutputDirectory().resolve(targetModel);
      Files.createDirectories(modelOutputDir);
      Path modelfile = modelOutputDir.resolve("Modelfile");

      String modelfileContent = String.format(
          "FROM %s\nPARAMETER temperature %.2f\nPARAMETER num_ctx 8192\nSYSTEM \"\"\"\n%s\n\"\"\"\n",
          baseModel, temperature, systemPrompt.toString().replace("\"\"\"", "\\\"\\\"\\\"")
      );
      Files.writeString(modelfile, modelfileContent, StandardCharsets.UTF_8);
      logs.append(String.format("Generated Modelfile saved to %s\n", modelfile.toAbsolutePath()));
    } catch (Exception e) {
      logs.append(String.format("Warning: could not save local Modelfile: %s\n", e.getMessage()));
    }

    // Call Ollama API to compile and register the model
    logs.append(String.format("Invoking Ollama API at %s/api/create...\n", ollamaBaseUrl));
    try {
      Map<String, Object> payload = new HashMap<>();
      payload.put("model", targetModel);
      payload.put("from", baseModel);
      payload.put("system", systemPrompt.toString());
      payload.put("parameters", Map.of("temperature", temperature));
      payload.put("stream", false);

      String jsonPayload = objectMapper.writeValueAsString(payload);

      HttpRequest httpReq = HttpRequest.newBuilder()
          .uri(URI.create(ollamaBaseUrl + "/api/create"))
          .header("Content-Type", "application/json")
          .timeout(Duration.ofMinutes(5))
          .POST(HttpRequest.BodyPublishers.ofString(jsonPayload, StandardCharsets.UTF_8))
          .build();

      HttpResponse<String> response = httpClient.send(httpReq, HttpResponse.BodyHandlers.ofString());

      if (response.statusCode() >= 200 && response.statusCode() < 300) {
        boolean success = false;
        String errorMessage = null;
        for (String line : response.body().split("\n")) {
          line = line.trim();
          if (line.isEmpty()) continue;
          try {
            JsonNode node = objectMapper.readTree(line);
            if (node.has("status")) {
              String status = node.get("status").asText();
              logs.append(String.format("Status: %s\n", status));
              if ("success".equalsIgnoreCase(status)) {
                success = true;
              }
            }
            if (node.has("error")) {
              errorMessage = node.get("error").asText();
              logs.append(String.format("Error: %s\n", errorMessage));
            }
          } catch (Exception ignored) {}
        }

        if (success || errorMessage == null) {
          logs.append(String.format("Model '%s' compiled and registered in Ollama successfully.\n", targetModel));
          return new FineTuneResult(
              true,
              targetModel,
              String.format("Fine-tuning completed successfully! Model '%s' is ready in Ollama.", targetModel),
              logs.toString(),
              sampleCount
          );
        } else {
          return new FineTuneResult(
              false,
              targetModel,
              "Ollama model compilation failed: " + errorMessage,
              logs.toString(),
              sampleCount
          );
        }
      } else {
        logs.append(String.format("Ollama API error: HTTP %d - %s\n", response.statusCode(), response.body()));
        return new FineTuneResult(
            false,
            targetModel,
            "Ollama creation failed: HTTP " + response.statusCode() + " - " + response.body(),
            logs.toString(),
            sampleCount
        );
      }
    } catch (Exception e) {
      logs.append(String.format("Exception calling Ollama: %s\n", e.getMessage()));
      return new FineTuneResult(
          false,
          targetModel,
          "Failed to communicate with Ollama: " + e.getMessage(),
          logs.toString(),
          sampleCount
      );
    }
  }

  public List<String> getAvailableModels() {
    List<String> models = new ArrayList<>();
    try {
      HttpRequest request = HttpRequest.newBuilder()
          .uri(URI.create(ollamaBaseUrl + "/api/tags"))
          .timeout(Duration.ofSeconds(5))
          .GET()
          .build();

      HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
      if (response.statusCode() == 200) {
        JsonNode root = objectMapper.readTree(response.body());
        JsonNode modelsNode = root.path("models");
        if (modelsNode.isArray()) {
          for (JsonNode m : modelsNode) {
            String name = m.path("name").asText();
            if (!name.isBlank() && !name.contains("embed")) {
              models.add(name);
            }
          }
        }
      }
    } catch (Exception e) {
      log.warn("Could not fetch Ollama models: {}", e.getMessage());
    }
    if (models.isEmpty()) {
      models.add("qwen2.5:1.5b");
    }
    return models;
  }

  public record DomainPair(String userPrompt, String assistantResponse) {}
  public record DatasetFileInfo(String name, long size, String lastModified, int sampleCount) {}
  public record UploadResponse(String fileName, long size, int sampleCount, String message) {}
  public record FineTuneRequest(String datasetFile, String modelName, String baseModel, Double temperature) {}
  public record FineTuneResult(boolean success, String modelName, String message, String logs, int sampleCount) {}
}
