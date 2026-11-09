package com.example.ai.controller;

import com.example.ai.service.FineTuningLoraService;
import tools.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.util.concurrent.CompletableFuture;

@RestController
@RequestMapping("/api/lorafinetune")
public class FineTuningLoraController {
  private static final Logger log = LoggerFactory.getLogger(FineTuningLoraController.class);
  private final FineTuningLoraService fineTuningService;
  private final ObjectMapper objectMapper;

  public FineTuningLoraController(FineTuningLoraService fineTuningService, ObjectMapper objectMapper) {
    this.fineTuningService = fineTuningService;
    this.objectMapper = objectMapper;
  }

  @PostMapping("/loraprocess")
  public ResponseEntity<FineTuningLoraService.FineTuneResult> process(@RequestBody FineTuningLoraService.FineTuneRequest request) {
    FineTuningLoraService.FineTuneResult result = fineTuningService.processFineTuning(request);
    return ResponseEntity.ok(result);
  }

  @PostMapping(value = "/loraprocess-stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
  public SseEmitter processStream(@RequestBody FineTuningLoraService.FineTuneRequest request) {
    SseEmitter emitter = new SseEmitter(30 * 60 * 1000L); // 30 mins timeout for training
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

  @GetMapping(value = "/loraprocess-stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
  public SseEmitter processStreamGet(
      @RequestParam(value = "datasetFile", required = false) String datasetFile,
      @RequestParam(value = "modelName", required = false) String modelName,
      @RequestParam(value = "baseModel", required = false) String baseModel,
      @RequestParam(value = "temperature", required = false, defaultValue = "0.1") Double temperature) {
    return processStream(new FineTuningLoraService.FineTuneRequest(datasetFile, modelName, baseModel, temperature));
  }
}

