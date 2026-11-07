package com.example.ai.controller;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/chat")
public class ChatController {
  private final ChatClient chatClient;
  public ChatController(ChatClient chatClient) { this.chatClient = chatClient; }

  @PostMapping
  public ChatResponse chat(@RequestBody ChatRequest request) {
    String answer = chatClient.prompt().user(request.message()).call().content();
    return new ChatResponse(answer);
  }
  public record ChatRequest(String message) {}
  public record ChatResponse(String answer) {}
}
