package com.example.ai.controller;

import com.example.ai.service.RagService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

@RestController
@RequestMapping("/api/rag")
public class RagController {
  private final RagService ragService;
  public RagController(RagService ragService) { this.ragService = ragService; }

  @PostMapping("/text")
  public ResponseEntity<IngestResponse> ingest(@RequestBody IngestRequest request) {
    return ResponseEntity.ok(new IngestResponse(ragService.ingestText(request.text(), request.source())));
  }

  @PostMapping("/file")
  public ResponseEntity<IngestResponse> upload(@RequestParam("file") MultipartFile file) throws Exception {
    return ResponseEntity.ok(new IngestResponse(ragService.ingestFile(file)));
  }
  public record IngestRequest(String text, String source) {}
  public record IngestResponse(int documentsAdded) {}
}
