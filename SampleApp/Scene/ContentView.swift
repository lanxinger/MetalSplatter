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
                // Note: ARModelIdentifier removed - it was never used. AR navigation would use ModelIdentifier directly if needed.
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
            
            Text("• PLY, SPZ, SPLAT files: Use 'Read Scene File'\n• SOGS v2 bundled: Select .sog file via 'Read Scene File'\n• SOGS v1 folders: Use 'Read SOGS Folder'\n• SOGS fallback: Select meta.json directly via 'Read Scene File'")
                .font(.caption)
                .multilineTextAlignment(.leading)
                .padding(.horizontal)

            Spacer()

            Button("Read Scene File (PLY/SPLAT/SPZ/JSON/SOG/ZIP)") {
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
                            UTType(filenameExtension: "sog")!,
                            UTType.zip
                          ]) {
                isPickingFile = false
                switch $0 {
                case .success(let url):
                    // Start security-scoped access - ModelCache will manage the lifecycle
                    let hasAccess = url.startAccessingSecurityScopedResource()
                    let model = ModelIdentifier.gaussianSplat(url)

                    // Pre-load model into cache with security-scoped access tracking
                    // This ensures the access is released when the model is evicted
                    Task {
                        var accessReleased = false
                        defer {
                            // Ensure security scope is released if we didn't hand it off to cache
                            // This handles both errors and task cancellation
                            if !accessReleased && hasAccess {
                                url.stopAccessingSecurityScopedResource()
                            }
                        }

                        do {
                            // Check for cancellation before expensive operation
                            try Task.checkCancellation()

                            _ = try await ModelCache.shared.getModel(
                                model,
                                securityScopedURL: url,
                                hasSecurityScopedAccess: hasAccess
                            )
                            // Access successfully handed off to cache
                            accessReleased = true

                            await MainActor.run {
                                lastLoadedModel = model
                                openWindow(value: model)
                            }
                        } catch is CancellationError {
                            print("Model loading cancelled for: \(url.lastPathComponent)")
                            // accessReleased is false, defer will clean up
                        } catch {
                            print("Failed to load model: \(error)")
                            // accessReleased is false, defer will clean up
                        }
                    }
                case .failure(let error):
                    print("File picker failed: \(error.localizedDescription)")
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
        let model = ModelIdentifier.gaussianSplat(metaURL)

        // Pre-load model into cache with security-scoped access tracking
        // ModelCache will manage the folder access lifecycle
        Task {
            var accessReleased = false
            defer {
                // Ensure security scope is released if we didn't hand it off to cache
                // This handles both errors and task cancellation
                if !accessReleased && hasAccess {
                    folderURL.stopAccessingSecurityScopedResource()
                }
            }

            do {
                // Check for cancellation before expensive operation
                try Task.checkCancellation()

                _ = try await ModelCache.shared.getModel(
                    model,
                    securityScopedURL: folderURL,  // Track folder URL, not meta.json
                    hasSecurityScopedAccess: hasAccess
                )
                // Access successfully handed off to cache
                accessReleased = true

                await MainActor.run {
                    lastLoadedModel = model
                    openWindow(value: model)
                }
            } catch is CancellationError {
                print("SOGS model loading cancelled for: \(folderURL.lastPathComponent)")
                // accessReleased is false, defer will clean up
            } catch {
                print("Failed to load SOGS model: \(error)")
                // accessReleased is false, defer will clean up
            }
        }
    }
}
