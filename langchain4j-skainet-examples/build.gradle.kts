plugins {
    kotlin("jvm") version "2.2.21"
}

kotlin {
    jvmToolchain(21)
}

dependencies {
    implementation(project(":langchain4j-skainet"))
    implementation("dev.langchain4j:langchain4j:${project.property("langchain4jVersion")}")
    implementation("ch.qos.logback:logback-classic:${project.property("logbackVersion")}")
}

val jvmArgs = listOf(
    "--add-modules=jdk.incubator.vector",
    "--enable-native-access=ALL-UNNAMED"
)

tasks.register<JavaExec>("runChat") {
    group = "examples"
    description = "Run the basic chat model example"
    classpath = sourceSets["main"].runtimeClasspath
    mainClass.set("sk.ainet.langchain4j.examples.SkainetChatModelExamplesKt")
    jvmArgs(jvmArgs)
}

tasks.register<JavaExec>("runStream") {
    group = "examples"
    description = "Run the streaming chat model example"
    classpath = sourceSets["main"].runtimeClasspath
    mainClass.set("sk.ainet.langchain4j.examples.SkainetStreamingChatModelExamplesKt")
    jvmArgs(jvmArgs)
}

tasks.register<JavaExec>("runFunctions") {
    group = "examples"
    description = "Run the function calling example"
    classpath = sourceSets["main"].runtimeClasspath
    mainClass.set("sk.ainet.langchain4j.examples.SkainetFunctionCallingExamplesKt")
    jvmArgs(jvmArgs)
}

tasks.register<JavaExec>("runRag") {
    group = "examples"
    description = "Run the RAG embedding example"
    classpath = sourceSets["main"].runtimeClasspath
    mainClass.set("sk.ainet.langchain4j.examples.SkainetBasicRagEmbedExamplesKt")
    jvmArgs(jvmArgs)
}
