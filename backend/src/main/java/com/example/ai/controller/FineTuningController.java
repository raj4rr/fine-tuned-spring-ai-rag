package com.example.ai.controller;

import com.example.ai.service.FineTuningService;
import tools.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.io.IOException;
import java.util.List;
import java.util.concurrent.CompletableFuture;

@RestController
@RequestMapping("/api/finetune")
public class FineTuningController {
  private static final Logger log = LoggerFactory.getLogger(FineTuningController.class);
  private final FineTuningService fineTuningService;
  private final ObjectMapper objectMapper;

  public FineTuningController(FineTuningService fineTuningService, ObjectMapper objectMapper) {
    this.fineTuningService = fineTuningService;
    this.objectMapper = objectMapper;
  }

  @GetMapping("/files")
  public ResponseEntity<List<FineTuningService.DatasetFileInfo>> listFiles() {
    return ResponseEntity.ok(fineTuningService.listDatasetFiles());
  }

  @PostMapping("/upload")
  public ResponseEntity<FineTuningService.UploadResponse> uploadFile(@RequestParam("file") MultipartFile file) {
    try {
      return ResponseEntity.ok(fineTuningService.uploadDataset(file));
    } catch (IllegalArgumentException e) {
      return ResponseEntity.badRequest().body(new FineTuningService.UploadResponse(null, 0, 0, e.getMessage()));
    } catch (IOException e) {
      return ResponseEntity.internalServerError().body(new FineTuningService.UploadResponse(null, 0, 0, "Upload error: " + e.getMessage()));
    }
  }

  @PostMapping("/process")
  public ResponseEntity<FineTuningService.FineTuneResult> process(@RequestBody FineTuningService.FineTuneRequest request) {
    FineTuningService.FineTuneResult result = fineTuningService.processFineTuning(request);
    return ResponseEntity.ok(result);
  }

  @PostMapping(value = "/process-stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
  public SseEmitter processStream(@RequestBody FineTuningService.FineTuneRequest request) {
    SseEmitter emitter = new SseEmitter(15 * 60 * 1000L); // 15 mins timeout
    CompletableFuture.runAsync(() -> {
      try {
        fineTuningService.processFineTuningStream(request, evt -> {
          try {
            String json = objectMapper.writeValueAsString(evt);
            emitter.send(SseEmitter.event().name(evt.type()).data(json));
          } catch (Exception e) {
            log.debug("SSE client disconnected: {}", e.getMessage());
          }
        });
      } catch (Exception e) {
        try {
          emitter.send(SseEmitter.event().name("error").data("{\"type\":\"error\",\"message\":\"" + e.getMessage() + "\"}"));
        } catch (Exception ignored) {}
      } finally {
        try {
          emitter.complete();
        } catch (Exception ignored) {}
      }
    });
    return emitter;
  }

  @GetMapping(value = "/process-stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
  public SseEmitter processStreamGet(
      @RequestParam(value = "datasetFile", required = false) String datasetFile,
      @RequestParam(value = "modelName", required = false) String modelName,
      @RequestParam(value = "baseModel", required = false) String baseModel,
      @RequestParam(value = "temperature", required = false, defaultValue = "0.1") Double temperature) {
    return processStream(new FineTuningService.FineTuneRequest(datasetFile, modelName, baseModel, temperature));
  }

  @GetMapping("/models")
  public ResponseEntity<List<String>> getModels() {
    return ResponseEntity.ok(fineTuningService.getAvailableModels());
  }
}

