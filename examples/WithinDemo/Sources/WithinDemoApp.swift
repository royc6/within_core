import SwiftUI

/// A minimal app on WithinCore: set up the model, chat with it, and measure it.
@main
struct WithinDemoApp: App {
    @State private var store = try? ModelStore()
    @State private var holder = EngineHolder()
    @Environment(\.scenePhase) private var scenePhase

    var body: some Scene {
        WindowGroup {
            if let store {
                TabView {
                    Tab("Chat", systemImage: "text.bubble") {
                        NavigationStack {
                            if store.state == .installed {
                                ChatScreen(store: store, holder: holder)
                            } else {
                                ContentUnavailableView("Set up the model first", systemImage: "arrow.down.circle")
                            }
                        }
                    }
                    Tab("Benchmark", systemImage: "speedometer") {
                        NavigationStack {
                            if store.state == .installed {
                                BenchmarkScreen(store: store, holder: holder)
                            } else {
                                ContentUnavailableView("Set up the model first", systemImage: "arrow.down.circle")
                            }
                        }
                    }
                    Tab("Model", systemImage: "internaldrive") {
                        NavigationStack { ModelScreen(store: store) }
                    }
                }
                // In the background the system may stop the app for its memory; release the model first.
                .onChange(of: scenePhase) { _, phase in
                    if phase == .background { holder.release() }
                }
                // `-benchmark` and `-chat "<prompt>"` run without taps (LaunchCommand).
                .task { await LaunchCommand.requested?.run(store: store, holder: holder) }
            } else {
                ContentUnavailableView("The bundled model catalog is missing", systemImage: "exclamationmark.triangle")
            }
        }
    }
}
