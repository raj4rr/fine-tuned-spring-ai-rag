package com.example.ai.service;

import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;
import org.springframework.ai.document.Document;
import org.springframework.ai.document.DocumentReader;
import org.springframework.ai.reader.TextReader;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

@Service
public class RagService {
  private final VectorStore vectorStore;
  public RagService(VectorStore vectorStore) { this.vectorStore = vectorStore; }

  public int ingestText(String text, String source) {
    List<Document> docs = List.of(new Document(text, Map.of("source", source == null ? "api" : source)));
    vectorStore.add(docs);
    return docs.size();
  }

  public int ingestFile(MultipartFile file) throws Exception {
    return ingestText(new String(file.getBytes(), StandardCharsets.UTF_8), file.getOriginalFilename());
  }
}
