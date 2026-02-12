package sk.ainet.langchain4j.examples

import dev.langchain4j.agent.tool.P
import dev.langchain4j.agent.tool.Tool
import dev.langchain4j.memory.chat.MessageWindowChatMemory
import dev.langchain4j.model.chat.ChatModel
import dev.langchain4j.service.AiServices
import dev.langchain4j.service.SystemMessage
import sk.ainet.langchain4j.SkainetChatModel
import java.nio.file.Path

/**
 * Function calling example using SKaiNET as the LLM backend.
 * Mirrors JlamaAiFunctionCallingExamples.java.
 */
fun main() {
    val model: ChatModel = SkainetChatModel.builder()
        .modelPath(Path.of("models/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf"))
        .temperature(0.0f)
        .maxTokens(512)
        .build()

    val paymentTool = PaymentTransactionTool()
    val userMessage = "What is the status and the payment date of transaction T1005?"

    val agent = AiServices.builder(Assistant::class.java)
        .chatModel(model)
        .tools(paymentTool)
        .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
        .build()

    val answer = agent.chat(userMessage)
    println(answer)
}

interface Assistant {
    @SystemMessage(
        "You are a payment transaction support agent.",
        "You MUST use the payment transaction tool to search the payment transaction data.",
        "If there is a date, convert it in a human readable format."
    )
    fun chat(userMessage: String): String
}

class PaymentTransactionTool {

    @Tool("Get payment status of a transaction")
    fun retrievePaymentStatus(
        @P("Transaction id to search payment data") transactionId: String
    ): String = getPaymentDataField(transactionId, "payment_status")

    @Tool("Get payment date of a transaction")
    fun retrievePaymentDate(
        @P("Transaction id to search payment data") transactionId: String
    ): String = getPaymentDataField(transactionId, "payment_date")

    private fun getPaymentData(): Map<String, List<String>> = mapOf(
        "transaction_id" to listOf("T1001", "T1002", "T1003", "T1004", "T1005"),
        "customer_id" to listOf("C001", "C002", "C003", "C002", "C001"),
        "payment_amount" to listOf("125.50", "89.99", "120.00", "54.30", "210.20"),
        "payment_date" to listOf("2021-10-05", "2021-10-06", "2021-10-07", "2021-10-05", "2021-10-08"),
        "payment_status" to listOf("Paid", "Unpaid", "Paid", "Paid", "Pending")
    )

    private fun getPaymentDataField(transactionId: String, data: String): String {
        val transactionIds = getPaymentData()["transaction_id"]!!
        val paymentData = getPaymentData()[data]!!
        val index = transactionIds.indexOf(transactionId)
        return if (index != -1) paymentData[index] else "Transaction ID not found"
    }
}
