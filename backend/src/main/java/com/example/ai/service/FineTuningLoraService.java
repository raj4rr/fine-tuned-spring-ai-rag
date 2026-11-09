package com.example.ai.service;


import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

@Service
public class FineTuningLoraService {
    private static final Logger log = LoggerFactory.getLogger(FineTuningLoraService.class);

    private static String nowFormatted() {
        return java.time.LocalTime.now().format(java.time.format.DateTimeFormatter.ofPattern("HH:mm:ss"));
    }

    public FineTuneResult processFineTuning(FineTuneRequest request) {
        return processFineTuningStream(request, evt -> {
            if ("log".equals(evt.type())) {
                log.info("[LoRA FineTune] [{}] {}", evt.level(), evt.message());
            }
        });
    }

    public FineTuneResult processFineTuningStream(FineTuneRequest request, java.util.function.Consumer<FineTuningService.FineTuneStreamEvent> emitter) {
        StringBuilder logs = new StringBuilder();
        emitter.accept(new FineTuningService.FineTuneStreamEvent("log", "STAGE", nowFormatted(), "🚀 Initializing LoRA / QLoRA training pipeline...", null, null, null));

        Path datasetPath = getDatasetDirectory().resolve(request.datasetFile());
        Path outputPath = getOutputDirectory().resolve(request.modelName());
        Path ggufPath = outputPath.resolve("gguf");

        Path current = Paths.get(".").toAbsolutePath().normalize();
        Path projectRoot = current.endsWith("backend") ? current.getParent() : current;
        Path venvPython = projectRoot.resolve(".venv").resolve("bin").resolve("python3");
        String pythonCmd = Files.isExecutable(venvPython) ? venvPython.toAbsolutePath().toString() : "python3";

        Path scriptPath = projectRoot.resolve("fine-tuning").resolve("trainmac.py");
        if (!Files.exists(scriptPath)) {
            scriptPath = projectRoot.resolve("fine-tuning").resolve("train.py");
        }

        emitter.accept(new FineTuningService.FineTuneStreamEvent("log", "INFO", nowFormatted(), "🐍 Python Interpreter: " + pythonCmd, null, null, null));
        emitter.accept(new FineTuningService.FineTuneStreamEvent("log", "INFO", nowFormatted(), "📜 Training Script: " + scriptPath.toAbsolutePath(), null, null, null));
        emitter.accept(new FineTuningService.FineTuneStreamEvent("log", "INFO", nowFormatted(), "📁 Dataset: " + datasetPath.toAbsolutePath(), null, null, null));
        emitter.accept(new FineTuningService.FineTuneStreamEvent("log", "INFO", nowFormatted(), "🎯 Target Model: " + request.modelName(), null, null, null));
        emitter.accept(new FineTuningService.FineTuneStreamEvent("log", "STAGE", nowFormatted(), "⚡ Launching Python process...", null, null, null));

        try {
            // 1. Prepare environment and execution command
            ProcessBuilder pb = new ProcessBuilder(
                    pythonCmd, scriptPath.toAbsolutePath().toString(),
                    "--dataset", datasetPath.toAbsolutePath().toString(),
                    "--output", outputPath.toAbsolutePath().toString(),
                    "--gguf", ggufPath.toAbsolutePath().toString()
            );

            pb.redirectErrorStream(true);
            Process process = pb.start();

            // 2. Stream Python training logs back in real-time
            try (BufferedReader reader = new BufferedReader(
                    new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    logs.append(line).append("\n");
                    log.info("[Python Train] {}", line);
                    emitter.accept(new FineTuningService.FineTuneStreamEvent("log", "PYTHON", nowFormatted(), line, null, null, null));
                }
            }

            int exitCode = process.waitFor();
            if (exitCode != 0) {
                String failMsg = "Python training failed with exit code: " + exitCode;
                emitter.accept(new FineTuningService.FineTuneStreamEvent("log", "ERROR", nowFormatted(), "❌ " + failMsg, false, request.modelName(), 0));
                emitter.accept(new FineTuningService.FineTuneStreamEvent("error", "ERROR", nowFormatted(), failMsg, false, request.modelName(), 0));
                return new FineTuneResult(false, request.modelName(), failMsg, logs.toString(), 0);
            }

            // 3. Register the compiled GGUF file into Ollama natively
            emitter.accept(new FineTuningService.FineTuneStreamEvent("log", "STAGE", nowFormatted(), "📦 Training successful! Registering GGUF with Ollama...", null, null, null));
            Path modelfilePath = outputPath.resolve("Modelfile");
            String modelfileContent = String.format("FROM %s\n", ggufPath.resolve("unsloth.Q4_K_M.gguf").toAbsolutePath());
            Files.writeString(modelfilePath, modelfileContent);
            emitter.accept(new FineTuningService.FineTuneStreamEvent("log", "INFO", nowFormatted(), "Created Modelfile at " + modelfilePath.toAbsolutePath(), null, null, null));

            String doneMsg = "True LoRA / QLoRA Fine-tuning completed successfully!";
            emitter.accept(new FineTuningService.FineTuneStreamEvent("log", "SUCCESS", nowFormatted(), "🎉 " + doneMsg, true, request.modelName(), 0));
            emitter.accept(new FineTuningService.FineTuneStreamEvent("complete", "SUCCESS", nowFormatted(), doneMsg, true, request.modelName(), 0));
            return new FineTuneResult(true, request.modelName(), doneMsg, logs.toString(), 0);

        } catch (Exception e) {
            String exMsg = "Exception during execution: " + e.getMessage();
            logs.append(exMsg);
            emitter.accept(new FineTuningService.FineTuneStreamEvent("log", "ERROR", nowFormatted(), "❌ " + exMsg, false, request.modelName(), 0));
            emitter.accept(new FineTuningService.FineTuneStreamEvent("error", "ERROR", nowFormatted(), exMsg, false, request.modelName(), 0));
            return new FineTuneResult(false, request.modelName(), exMsg, logs.toString(), 0);
        }
    }


    public record DomainPair(String userPrompt, String assistantResponse) {}
    public record DatasetFileInfo(String name, long size, String lastModified, int sampleCount) {}
    public record UploadResponse(String fileName, long size, int sampleCount, String message) {}
    public record FineTuneRequest(String datasetFile, String modelName, String baseModel, Double temperature) {}
    public record FineTuneResult(boolean success, String modelName, String message, String logs, int sampleCount) {}

    public Path getDatasetDirectory() {
        Path current = Paths.get(".").toAbsolutePath().normalize();
        Path projectRoot = current;
        if (current.endsWith("backend")) {
            projectRoot = current.getParent();
        }

        Path datasetDir = projectRoot.resolve("fine-tuning").resolve("dataset");
        if (!Files.exists(datasetDir)) {
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

    public List<FineTuningService.DatasetFileInfo> listDatasetFiles() {
        Path dir = getDatasetDirectory();
        List<FineTuningService.DatasetFileInfo> files = new ArrayList<>();
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
                files.add(new FineTuningService.DatasetFileInfo(p.getFileName().toString(), size, modified.toString(), lineCount));
            }
        } catch (IOException e) {
            log.error("Error reading dataset directory", e);
        }
        return files;
    }

}
