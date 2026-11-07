package com.example.ai.controller;

import com.example.ai.service.FineTuningService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.List;

@RestController
@RequestMapping("/api/finetune")
public class FineTuningController {
  private final FineTuningService fineTuningService;

  public FineTuningController(FineTuningService fineTuningService) {
    this.fineTuningService = fineTuningService;
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

  @GetMapping("/models")
  public ResponseEntity<List<String>> getModels() {
    return ResponseEntity.ok(fineTuningService.getAvailableModels());
  }
}

