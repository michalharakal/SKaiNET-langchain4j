package sk.ainet.langchain4j.examples

import dev.langchain4j.data.message.SystemMessage
import dev.langchain4j.data.message.UserMessage
import dev.langchain4j.model.chat.ChatModel
import dev.langchain4j.model.chat.response.ChatResponse
import sk.ainet.langchain4j.SkainetChatModel
import java.nio.file.Path

/**
 * Basic chat example using SKaiNET as the LLM backend.
 * Mirrors JlamaChatModelExamples.java.
 */
fun main() {
    val model: ChatModel = SkainetChatModel.builder()
        .modelPath(Path.of("models/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf"))
        .temperature(0.3f)
        .maxTokens(512)
        .build()

    val chatResponse: ChatResponse = model.chat(
        SystemMessage.from("You are helpful chatbot who is a java expert."),
        UserMessage.from("Write a java program to print hello world.")
    )

    println("\n${chatResponse.aiMessage().text()}\n")
}
