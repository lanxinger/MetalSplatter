import Foundation

#if os(iOS)

struct ARModelIdentifier: Hashable {
    let model: ModelIdentifier?
    
    init(model: ModelIdentifier?) {
        self.model = model
    }
}

#endif // os(iOS)