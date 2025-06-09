import SwiftUI
import RealityKit
import UniformTypeIdentifiers

struct ContentView: View {
    @State private var isPickingFile = false
    @State private var isPickingSOGSFolder = false

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

            Button("Read Scene File") {
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
                            UTType(filenameExtension: "json")!
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
                    openWindow(value: ModelIdentifier.gaussianSplat(url))
                case .failure:
                    break
                }
            }
            
            Button("Read SOGS Folder") {
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
        // Keep folder access for longer since SOGS needs multiple files
        Task {
            try await Task.sleep(for: .seconds(60)) // Longer timeout for SOGS loading
            if hasAccess {
                print("Stopping security scoped access for folder")
                folderURL.stopAccessingSecurityScopedResource()
            }
        }
        
        openWindow(value: ModelIdentifier.gaussianSplat(metaURL))
    }
}
