// New for TQF (not derived from NVMAI's own RootView): composes the
// adopted NVMAIMacTheme/HUDMetricView/ResponseMarkdownRenderer pieces
// with a TQF-specific prompt/output layout driven by TqfAppModel.

import AppKit
import SwiftUI

public struct RootView: View {
    @Bindable var model: TqfAppModel

    public init(model: TqfAppModel) {
        self.model = model
    }

    public var body: some View {
        HStack(spacing: 0) {
            VStack(spacing: 0) {
                header
                Divider()
                outputPane
                Divider()
                composer
            }
            .frame(minWidth: 640, minHeight: 480)
            if model.showsInspector {
                Divider()
                InspectorView(metrics: model.metrics)
                    .padding(12)
                    .transition(.move(edge: .trailing))
            }
        }
        .animation(.default, value: model.showsInspector)
        .task {
            await model.refreshServerStatus()
            model.startMetricsPolling()
        }
        .onDisappear { model.stopMetricsPolling() }
    }

    private var header: some View {
        HStack {
            Circle()
                .fill(model.serverReachable ? NVMAIMacTheme.accentColor : Color.red)
                .frame(width: 8, height: 8)
            Text(model.serverReachable ? "TurboQwenFare — connected" : "TurboQwenFare — server unreachable")
                .font(.headline)
            Spacer()
            if let error = model.error {
                Text(error)
                    .font(.caption)
                    .foregroundStyle(.red)
                    .lineLimit(1)
            }
            // spec §47: "expandable engineering cockpit" — the simple
            // default view (everything else in this file) never
            // changes; this toggle only reveals/hides InspectorView.
            Button {
                model.showsInspector.toggle()
            } label: {
                Label("Inspector", systemImage: "sidebar.trailing")
                    .labelStyle(.iconOnly)
            }
            .buttonStyle(.plain)
            .help("Toggle the engineering inspector")
        }
        .padding(12)
    }

    private var outputPane: some View {
        ScrollView {
            Text(AttributedString(
                ResponseMarkdownRenderer().render(model.outputText.isEmpty ? " " : model.outputText)
                    .attributedString
            ))
            .textSelection(.enabled)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(12)
        }
    }

    private var composer: some View {
        HStack(alignment: .bottom, spacing: 8) {
            TextEditor(text: $model.promptText)
                .frame(minHeight: 36, maxHeight: 120)
                .font(.body)
                .overlay(alignment: .topLeading) {
                    if model.promptText.isEmpty {
                        Text("Ask TurboQwenFare…")
                            .foregroundStyle(.tertiary)
                            .padding(.top, 8)
                            .padding(.leading, 5)
                            .allowsHitTesting(false)
                    }
                }
            if model.isGenerating {
                Button("Cancel", action: model.cancel)
            } else {
                Button("Send", action: model.generate)
                    .keyboardShortcut(.return, modifiers: .command)
                    .disabled(!model.canGenerate)
                    .tint(NVMAIMacTheme.accentColor)
            }
        }
        .padding(12)
    }
}
