plugins {
    kotlin("jvm") version "2.2.21"
    kotlin("plugin.serialization") version "2.2.21"
}

kotlin {
    jvmToolchain(21)
}

dependencies {
    // LangChain4j core API
    api("dev.langchain4j:langchain4j-core:${project.property("langchain4jVersion")}")

    // SKaiNET modules (resolved via composite build)
    implementation("sk.ainet:skainet-kllama")
    implementation("sk.ainet:skainet-kllama-agent")
    implementation("sk.ainet:skainet-bert")
    implementation("sk.ainet:skainet-llm")
    implementation("sk.ainet:skainet-backend-cpu")
    implementation("sk.ainet:skainet-lang-core")
    implementation("sk.ainet:skainet-compile-core")
    implementation("sk.ainet:skainet-io-core")
    implementation("sk.ainet:skainet-io-gguf")
    implementation("sk.ainet:skainet-io-safetensors")

    // Kotlin coroutines & serialization
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.10.2")
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.9.0")
}
