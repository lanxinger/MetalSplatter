import SwiftUI
import RealityKit
import UniformTypeIdentifiers

struct ContentView: View {
    @State private var isPickingFile = false
    @State private var isPickingSOGSFolder = false
    @State private var lastLoadedModel: ModelIdentifier?

#if os(macOS)
    @Environment(\.openWindow) private var openWindow
#elseif os(iOS)
    @State private var navigationPath = NavigationPath()

    private func openWindow(value: ModelIdentifier) {
        navigationPath.append(value)
    }
#elseif os(visionOS)
    @Environment(\.openImmersiveSpace) var openImmersiveSpace
    @Environment(\.dismissImmersiveSpace) var dismissImmersiveSpace

    @State var immersiveSpaceIsShown = false

    private func openWindow(value: ModelIdentifier) {
        Task {
            switch await openImmersiveSpace(value: value) {
            case .opened:
                immersiveSpaceIsShown = true
            case .error, .userCancelled:
                break
            @unknown default:
                break
            }
        }
    }
#endif

    var body: some View {
#if os(macOS) || os(visionOS)
        mainView
#elseif os(iOS)
        NavigationStack(path: $navigationPath) {
            mainView
                .navigationDestination(for: ModelIdentifier.self) { modelIdentifier in
                    MetalKitSceneView(modelIdentifier: modelIdentifier)
                        .navigationTitle(modelIdentifier.description)
                }
                .navigationDestination(for: ARModelIdentifier.self) { arModelIdentifier in
                    ARContentView(model: arModelIdentifier.model)
                        .navigationTitle("AR \(arModelIdentifier.model?.description ?? "View")")
                }
        }
#endif // os(iOS)
    }

    @ViewBuilder
    var mainView: some View {
        VStack {
            Spacer()

            Text("MetalSplatter SampleApp")
                .font(.title)
                .padding()

            Text("Load 3D Gaussian Splat Data")
                .font(.headline)
                .padding(.top)
            
            Text("• PLY, SPZ, SPLAT files: Use 'Read Scene File'\n• SOGS compressed: Use 'Read SOGS Folder'\n• SOGS fallback: Select meta.json directly via 'Read Scene File'")
                .font(.caption)
                .multilineTextAlignment(.leading)
                .padding(.horizontal)

            Spacer()

            Button("Read Scene File (PLY/SPLAT/SPZ/JSON/ZIP)") {
                isPickingFile = true
            }
            .padding()
            .buttonStyle(.borderedProminent)
            .disabled(isPickingFile)
#if os(visionOS)
            .disabled(immersiveSpaceIsShown)
#endif
            .fileImporter(isPresented: $isPickingFile,
                          allowedContentTypes: [
                            UTType(filenameExtension: "ply")!,
                            UTType(filenameExtension: "splat")!,
                            UTType(filenameExtension: "spz")!,
                            UTType(filenameExtension: "spx")!,
                            UTType(filenameExtension: "json")!,
                            UTType.zip
                          ]) {
                isPickingFile = false
                switch $0 {
                case .success(let url):
                    _ = url.startAccessingSecurityScopedResource()
                    Task {
                        // This is a sample app. In a real app, this should be more tightly scoped, not using a silly timer.
                        try await Task.sleep(for: .seconds(10))
                        url.stopAccessingSecurityScopedResource()
                    }
                    let model = ModelIdentifier.gaussianSplat(url)
                    lastLoadedModel = model
                    openWindow(value: model)
                case .failure:
                    break
                }
            }
            
            Button("Read SOGS Folder (or use ZIP above)") {
                isPickingSOGSFolder = true
            }
            .padding()
            .buttonStyle(.bordered)
            .disabled(isPickingSOGSFolder)
#if os(visionOS)
            .disabled(immersiveSpaceIsShown)
#endif
            .fileImporter(isPresented: $isPickingSOGSFolder,
                          allowedContentTypes: [UTType.folder]) { result in
                isPickingSOGSFolder = false
                
                // Handle the result safely with a small delay to avoid file picker crashes
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.1, execute: {
                    switch result {
                    case .success(let folderURL):
                        self.handleSOGSFolderSelection(folderURL)
                    case .failure(let error):
                        print("SOGS folder picker failed: \(error)")
                    }
                })
            }

            Spacer()

#if os(iOS)
            Button("Open in AR") {
                // Use the most recently loaded model, or default to sample box
                if let lastModel = lastLoadedModel {
                    navigationPath.append(ARModelIdentifier(model: lastModel))
                } else {
                    // Default to sample box if no models have been loaded
                    navigationPath.append(ARModelIdentifier(model: .sampleBox))
                }
            }
            .padding()
            .buttonStyle(.bordered)
            .opacity(lastLoadedModel != nil ? 1.0 : 0.7)

            Spacer()
#endif

            Button("Show Sample Box") {
                openWindow(value: ModelIdentifier.sampleBox)
            }
            .padding()
            .buttonStyle(.borderedProminent)
#if os(visionOS)
            .disabled(immersiveSpaceIsShown)
#endif

            Spacer()

#if os(visionOS)
            Button("Dismiss Immersive Space") {
                Task {
                    await dismissImmersiveSpace()
                    immersiveSpaceIsShown = false
                }
            }
            .disabled(!immersiveSpaceIsShown)

            Spacer()
#endif // os(visionOS)
        }
    }
    
    private func handleSOGSFolderSelection(_ folderURL: URL) {
        print("Selected SOGS folder: \(folderURL.path)")
        
        // Start accessing the folder with enhanced error handling
        let hasAccess = folderURL.startAccessingSecurityScopedResource()
        print("Security scoped access started: \(hasAccess)")
        
        // Look for meta.json inside the folder
        let metaURL = folderURL.appendingPathComponent("meta.json")
        print("Looking for meta.json at: \(metaURL.path)")
        
        // Check if meta.json exists with enhanced error handling
        var metaExists = false
        do {
            metaExists = FileManager.default.fileExists(atPath: metaURL.path)
            print("meta.json exists: \(metaExists)")
            
            if !metaExists {
                // Also try looking for any .json files in the folder
                let contents = try FileManager.default.contentsOfDirectory(at: folderURL, includingPropertiesForKeys: nil)
                let jsonFiles = contents.filter { $0.pathExtension.lowercased() == "json" }
                print("Found JSON files in folder: \(jsonFiles.map { $0.lastPathComponent })")
                
                if let firstJsonFile = jsonFiles.first {
                    print("Using first JSON file as meta file: \(firstJsonFile.path)")
                    handleSOGSSuccess(firstJsonFile, folderURL: folderURL, hasAccess: hasAccess)
                    return
                }
            }
        } catch {
            print("Error checking folder contents: \(error)")
        }
        
        if metaExists {
            print("Found SOGS meta.json, proceeding to load")
            handleSOGSSuccess(metaURL, folderURL: folderURL, hasAccess: hasAccess)
        } else {
            print("No meta.json found in selected folder")
            if hasAccess {
                folderURL.stopAccessingSecurityScopedResource()
            }
        }
    }
    
    private func handleSOGSSuccess(_ metaURL: URL, folderURL: URL, hasAccess: Bool) {
        // For SOGS, we need to maintain folder access until the model is loaded
        // Store the folder URL with the model identifier so the renderer can access it
        let model = ModelIdentifier.gaussianSplat(metaURL)
        lastLoadedModel = model
        openWindow(value: model)
        
        // Keep folder access for much longer since SOGS loading happens asynchronously
        Task {
            try await Task.sleep(for: .seconds(300)) // 5 minutes timeout for SOGS loading
            if hasAccess {
                print("Stopping security scoped access for folder after timeout")
                folderURL.stopAccessingSecurityScopedResource()
            }
        }
    }
}
