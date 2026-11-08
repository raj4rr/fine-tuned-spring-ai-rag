package com.example.ai.controller;

import com.example.ai.service.FineTuningLoraService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/lorafinetune")
public class FineTuningLoraController {
  private final FineTuningLoraService fineTuningService;

  public FineTuningLoraController(FineTuningLoraService fineTuningService) {
    this.fineTuningService = fineTuningService;
  }

  @PostMapping("/loraprocess")
  public ResponseEntity<FineTuningLoraService.FineTuneResult> process(@RequestBody FineTuningLoraService.FineTuneRequest request) {
    FineTuningLoraService.FineTuneResult result = fineTuningService.processFineTuning(request);
    return ResponseEntity.ok(result);
  }
}

