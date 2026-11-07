package com.example.ai.controller;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.advisor.vectorstore.QuestionAnswerAdvisor;
import org.springframework.ai.chat.prompt.PromptTemplate;
import org.springframework.ai.ollama.api.OllamaChatOptions;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/chat")
public class ChatController {
  private final ChatClient chatClient;
  private final QuestionAnswerAdvisor ragAdvisor;

  public ChatController(ChatClient chatClient, VectorStore vectorStore) {
    this.chatClient = chatClient;
    String customRagTemplate = """
        {query}

        Context information from knowledge base:
        ---------------------
        {question_answer_context}
        ---------------------

        Answer the query using the context above if relevant.
        If the context does not contain the answer, answer using your fine-tuned domain knowledge.
        """;
    this.ragAdvisor = QuestionAnswerAdvisor.builder(vectorStore)
        .promptTemplate(new PromptTemplate(customRagTemplate))
        .build();
  }

  @PostMapping
  public ChatResponse chat(@RequestBody ChatRequest request) {
    if (request.message() == null || request.message().isBlank()) {
      return new ChatResponse("Please enter a question or message.", request.model(), false);
    }

    var promptSpec = chatClient.prompt().user(request.message());
    String modelUsed = "default";

    if (request.model() != null && !request.model().isBlank() && !request.model().equalsIgnoreCase("default")) {
      promptSpec.options(OllamaChatOptions.builder().model(request.model().trim()));
      modelUsed = request.model().trim();
    } else {
      promptSpec.system("You are a senior Java, Spring AI and enterprise AI engineer.");
    }

    boolean ragActive = request.enableRag() == null || request.enableRag();
    if (ragActive) {
      promptSpec.advisors(this.ragAdvisor);
    }

    try {
      String answer = promptSpec.call().content();
      return new ChatResponse(answer, modelUsed, ragActive);
    } catch (Exception e) {
      return new ChatResponse("Error querying model [" + modelUsed + "]: " + e.getMessage(), modelUsed, ragActive);
    }
  }

  public record ChatRequest(String message, String model, Boolean enableRag) {}
  public record ChatResponse(String answer, String modelUsed, boolean ragEnabled) {}
}
