package sk.ainet.langchain4j.examples

import dev.langchain4j.data.message.ChatMessage
import dev.langchain4j.data.message.SystemMessage
import dev.langchain4j.data.message.UserMessage
import dev.langchain4j.model.chat.StreamingChatModel
import dev.langchain4j.model.chat.response.ChatResponse
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler
import sk.ainet.langchain4j.SkainetStreamingChatModel
import java.nio.file.Path
import java.util.concurrent.CompletableFuture

/**
 * Streaming chat example using SKaiNET as the LLM backend.
 * Mirrors JlamaStreamingChatModelExamples.java.
 */
fun main() {
    val futureResponse = CompletableFuture<ChatResponse>()

    val model: StreamingChatModel = SkainetStreamingChatModel.builder()
        .modelPath(Path.of("models/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf"))
        .temperature(0.3f)
        .maxTokens(256)
        .build()

    val messages: List<ChatMessage> = listOf(
        SystemMessage.from("You are a helpful chatbot that answers questions in under 30 words."),
        UserMessage.from("What is the best part of France and why?")
    )

    model.chat(messages, object : StreamingChatResponseHandler {
        override fun onPartialResponse(partialResponse: String) {
            print(partialResponse)
        }

        override fun onCompleteResponse(completeResponse: ChatResponse) {
            futureResponse.complete(completeResponse)
        }

        override fun onError(error: Throwable) {
            futureResponse.completeExceptionally(error)
        }
    })

    futureResponse.join()
    println()
}
