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
    private static final Logger log = LoggerFactory.getLogger(FineTuningService.class);

    // Keep your getDatasetDirectory() and getOutputDirectory() methods from before

    public FineTuneResult processFineTuning(FineTuneRequest request) {
        StringBuilder logs = new StringBuilder();
        logs.append(String.format("Starting true QLoRA fine-tuning via Python at %s\n", Instant.now()));

        Path datasetPath = getDatasetDirectory().resolve(request.datasetFile());
        Path outputPath = getOutputDirectory().resolve(request.modelName());
        Path ggufPath = outputPath.resolve("gguf");

        try {
            // 1. Prepare environment and execution command
            // Assuming your python script is saved as 'train.py' in your project root
            ProcessBuilder pb = new ProcessBuilder(
                    "python3", "/Users/rajesh/Desktop/spring-ai-rag-demo/backend/src/main/java/trainmac.py",
                    "--dataset", datasetPath.toAbsolutePath().toString(),
                    "--output", outputPath.toAbsolutePath().toString(),
                    "--gguf", ggufPath.toAbsolutePath().toString()
            );

            pb.redirectErrorStream(true);
            logs.append("Executing Unsloth QLoRA script on CUDA machine...\n");

            Process process = pb.start();

            // 2. Stream Python training logs back to Java in real-time
            try (BufferedReader reader = new BufferedReader(
                    new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    logs.append(line).append("\n");
                    log.info("[Python Train] {}", line); // Print to Java logs
                }
            }

            int exitCode = process.waitFor();
            if (exitCode != 0) {
                return new FineTuneResult(false, request.modelName(), "Python training failed.", logs.toString(), 0);
            }

            // 3. Register the compiled GGUF file into Ollama natively
            logs.append("Training successful! Registering GGUF with Ollama...\n");
            Path modelfilePath = outputPath.resolve("Modelfile");
            String modelfileContent = String.format("FROM %s\n", ggufPath.resolve("unsloth.Q4_K_M.gguf").toAbsolutePath());
            Files.writeString(modelfilePath, modelfileContent);

            // Call your existing Ollama API '/api/create' block here, pointing to this Modelfile
            return new FineTuneResult(true, request.modelName(), "True QLoRA Fine-tuning completed!", logs.toString(), 0);

        } catch (Exception e) {
            logs.append("Exception during execution: ").append(e.getMessage());
            return new FineTuneResult(false, request.modelName(), "Execution failed: " + e.getMessage(), logs.toString(), 0);
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
