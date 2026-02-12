package sk.ainet.langchain4j.examples

import dev.langchain4j.data.document.Document
import dev.langchain4j.data.document.parser.TextDocumentParser
import dev.langchain4j.data.document.splitter.DocumentSplitters
import dev.langchain4j.data.embedding.Embedding
import dev.langchain4j.data.message.AiMessage
import dev.langchain4j.data.segment.TextSegment
import dev.langchain4j.model.chat.ChatModel
import dev.langchain4j.model.embedding.EmbeddingModel
import dev.langchain4j.model.input.Prompt
import dev.langchain4j.model.input.PromptTemplate
import dev.langchain4j.store.embedding.EmbeddingMatch
import dev.langchain4j.store.embedding.EmbeddingSearchRequest
import dev.langchain4j.store.embedding.EmbeddingStore
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore
import sk.ainet.langchain4j.SkainetChatModel
import sk.ainet.langchain4j.SkainetEmbeddingModel
import java.net.URISyntaxException
import java.nio.file.Path
import java.nio.file.Paths
import java.util.stream.Collectors.joining

/**
 * RAG (Retrieval-Augmented Generation) example using SKaiNET for both
 * embeddings and chat completion.
 * Mirrors JlamaBasicRagEmbedExamples.java.
 */
fun main() {
    // Load the document about the origin of the llama
    val document: Document = dev.langchain4j.data.document.loader.FileSystemDocumentLoader.loadDocument(
        toPath("example-files/story-about-origin-of-the-llama.txt"),
        TextDocumentParser()
    )

    // Split the document into smaller chunks for embedding
    val splitter = DocumentSplitters.recursive(200, 0)
    val segments: List<TextSegment> = splitter.split(document)

    // Create embeddings using SKaiNET's BERT runtime
    val embeddingModel: EmbeddingModel = SkainetEmbeddingModel.builder()
        .modelDir(Path.of("models/e5-small-v2"))
        .modelName("e5-small-v2")
        .build()

    val embeddings: List<Embedding> = embeddingModel.embedAll(segments).content()

    // Store embeddings in an in-memory vector store
    val embeddingStore: EmbeddingStore<TextSegment> = InMemoryEmbeddingStore()
    embeddingStore.addAll(embeddings, segments)

    // Embed the user's question
    val question = "Who create the llamas?"
    val questionEmbedding: Embedding = embeddingModel.embed(question).content()

    // Search for relevant context
    val embeddingSearchRequest = EmbeddingSearchRequest.builder()
        .queryEmbedding(questionEmbedding)
        .maxResults(3)
        .minScore(0.7)
        .build()
    val relevantEmbeddings: List<EmbeddingMatch<TextSegment>> =
        embeddingStore.search(embeddingSearchRequest).matches()

    // Build a context-augmented prompt
    val promptTemplate = PromptTemplate.from(
        "Context information is below.:\n"
                + "------------------\n"
                + "{{information}}\n"
                + "------------------\n"
                + "Given the context information and not prior knowledge, answer the query.\n"
                + "Query: {{question}}\n"
                + "Answer:"
    )
    val information = relevantEmbeddings.stream()
        .map { match -> match.embedded().text() }
        .collect(joining("\n\n"))

    val promptInputs = mapOf<String, Any>(
        "question" to question,
        "information" to information
    )
    val prompt: Prompt = promptTemplate.apply(promptInputs)

    // Generate the answer using SKaiNET's LLM runtime
    val chatModel: ChatModel = SkainetChatModel.builder()
        .modelPath(Path.of("models/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf"))
        .temperature(0.2f)
        .maxTokens(256)
        .build()

    val aiMessage: AiMessage = chatModel.chat(prompt.toUserMessage()).aiMessage()
    val answer = aiMessage.text()
    println(answer)
}

private fun toPath(fileName: String): Path {
    try {
        val fileUrl = object {}.javaClass.classLoader.getResource(fileName)
            ?: throw RuntimeException("Resource not found: $fileName")
        return Paths.get(fileUrl.toURI())
    } catch (e: URISyntaxException) {
        throw RuntimeException(e)
    }
}
