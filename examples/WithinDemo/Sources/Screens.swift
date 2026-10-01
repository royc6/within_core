import SwiftUI
import WithinCore
import WithinCoreBase

/// Sets up the model: a download the user starts, with cellular data off unless allowed.
struct ModelScreen: View {
    let store: ModelStore

    var body: some View {
        @Bindable var store = store
        Form {
            Section {
                LabeledContent("Model", value: "Gemma 4 E4B")
                LabeledContent("Revision", value: store.manifest.revision)
                LabeledContent("Context", value: "\(store.manifest.contextLimit) tokens")
                LabeledContent("Download", value: store.totalBytes.formatted(.byteCount(style: .file)))
            }
            Section {
                switch store.state {
                case .installed:
                    Label("Installed", systemImage: "checkmark.circle")
                case .downloading(let file, let fraction):
                    ProgressView(value: fraction) { Text(file).font(.footnote.monospaced()) }
                    Button("Stop", role: .cancel) { store.cancel() }
                case .checking(let file):
                    ProgressView { Text("Checking \(file)").font(.footnote.monospaced()) }
                case .notInstalled, .failed:
                    Toggle("Allow cellular data", isOn: $store.allowsCellular)
                    Button("Download") { store.download() }
                    if case .failed(let message) = store.state {
                        Text(message).foregroundStyle(.red).font(.footnote)
                    }
                }
            } footer: {
                Text(
                    "Files come from Hugging Face and are checked against the SHA-256 in the bundled catalog. Keep the app open while it downloads."
                )
            }
        }
        .navigationTitle("Model")
    }
}

/// A plain chat: each reply streams in as the model writes it. The thought channel stays off.
struct ChatScreen: View {
    let store: ModelStore
    let holder: EngineHolder
    @State private var turns: [ChatTurn] = []
    @State private var partial = ""
    @State private var draft = ""
    @State private var generation: Task<Void, Never>?
    @State private var error: String?
    @FocusState private var composing: Bool

    var body: some View {
        VStack(spacing: 0) {
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 14) {
                    ForEach(Array(turns.enumerated()), id: \.offset) { _, turn in
                        Text(turn.text)
                            .foregroundStyle(turn.role == .user ? .secondary : .primary)
                            .frame(maxWidth: .infinity, alignment: turn.role == .user ? .trailing : .leading)
                    }
                    if generation != nil {
                        Text(partial.isEmpty ? (holder.isLoading ? "Preparing the model…" : "…") : partial)
                    }
                    if let error { Text(error).foregroundStyle(.red).font(.footnote) }
                }
                .textSelection(.enabled)
                .padding()
            }
            // Dragging the transcript or tapping anywhere in it puts the keyboard away.
            .scrollDismissesKeyboard(.interactively)
            .contentShape(Rectangle())
            .onTapGesture { composing = false }
            HStack {
                TextField("Message", text: $draft, axis: .vertical).textFieldStyle(.roundedBorder)
                    .focused($composing)
                if generation == nil {
                    Button("Send", action: send).disabled(draft.trimmingCharacters(in: .whitespaces).isEmpty)
                } else {
                    Button("Stop") { generation?.cancel() }
                }
            }
            .padding()
        }
        .navigationTitle("Chat")
        .toolbar { Button("Clear") { turns = [] }.disabled(generation != nil) }
    }

    private func send() {
        turns.append(ChatTurn(role: .user, text: draft))
        draft = ""; partial = ""; error = nil
        let turns = turns
        generation = Task {
            defer { generation = nil }
            do {
                let engine = try await holder.engine(folder: store.folder)
                let options = Gemma4Engine.Options(temperature: 0.7, topK: 64, maxOutputTokens: 1024, thinking: false)
                let text = try await reply(from: engine, turns: turns, options: options) { partial = $0 }
                self.turns.append(ChatTurn(role: .assistant, text: text))
            } catch is CancellationError {
                if !partial.isEmpty { self.turns.append(ChatTurn(role: .assistant, text: partial)) }
            } catch {
                self.error = error.localizedDescription
            }
        }
    }
}

/// Runs the performance procedure and shows each run.
struct BenchmarkScreen: View {
    let store: ModelStore
    let holder: EngineHolder
    @State private var status: String?
    @State private var result: Benchmark.Result?
    @State private var error: String?
    @State private var job: Task<Void, Never>?

    var body: some View {
        Form {
            Section {
                Text(Benchmark.prompt).font(.footnote)
                Button(job == nil ? "Run" : "Stop") { job == nil ? start() : job?.cancel() }
                if let status { Text(status).foregroundStyle(.secondary) }
                if let error { Text(error).foregroundStyle(.red).font(.footnote) }
            } footer: {
                Text(
                    "3 warmups of 64 tokens, then 5 runs of 512 tokens, each waiting for a nominal thermal state. Speed uses the median interval before tokens 33–512."
                )
            }
            if let result {
                Section("Result") {
                    LabeledContent(
                        "Decode",
                        value: "\(result.medianTokensPerSecond.formatted(.number.precision(.fractionLength(2)))) tok/s")
                    LabeledContent("First token", value: "\(Int(result.medianFirstTokenMs.rounded())) ms")
                    LabeledContent(
                        "Peak footprint",
                        value: Int64(result.peakFootprintBytes).formatted(.byteCount(style: .memory)))
                }
                Section("Runs") {
                    ForEach(result.runs) { run in
                        LabeledContent(
                            "Run \(run.id)",
                            value:
                                "\(run.tokensPerSecond.formatted(.number.precision(.fractionLength(2)))) tok/s · \(Int(run.firstTokenMs.rounded())) ms"
                        )
                    }
                }
            }
        }
        .navigationTitle("Benchmark")
    }

    private func start() {
        result = nil; error = nil
        // A run takes minutes; a locked screen would send the app to the background.
        UIApplication.shared.isIdleTimerDisabled = true
        job = Task {
            defer { job = nil; status = nil; UIApplication.shared.isIdleTimerDisabled = false }
            do {
                status = "Preparing the model"
                let engine = try await holder.engine(folder: store.folder)
                result = try await Benchmark.run(engine: engine) { status = $0 }
            } catch is CancellationError {
            } catch {
                self.error = error.localizedDescription
            }
        }
    }
}
