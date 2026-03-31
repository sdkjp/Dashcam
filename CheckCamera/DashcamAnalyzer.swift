import SwiftUI
import Vision
import CoreML
import AVFoundation

// MARK: - AI解析＆ストリーミング受信エンジン
class DashcamAnalyzer: NSObject, ObservableObject, URLSessionDataDelegate {
    @Published var alertMessage: String = "待機中..."
    @Published var isAlerting: Bool = false
    
    private var yoloModel: VNCoreMLModel?
    private var emergencyVehicleFrames = 0
    private var lastAlertTime: Date = Date.distantPast
    
    // IPWebcam（古いスマホ）からの映像受信用
    private var session: URLSession?
    private var dataTask: URLSessionDataTask?
    
    override init() {
        super.init()
        setupAI()
    }
    
    // 1. AIモデル（変換した yolov8n.mlpackage）の読み込み
    private func setupAI() {
        // Xcodeに yolov8n.mlpackage を入れると自動的にクラスが生成されるのでそれを読み込む
        // ※「yolov8n」の部分は実際のファイル名に合わせて変更します
        if let config = MLModelConfiguration() as? MLModelConfiguration,
           let mlModel = try? yolov8n(configuration: config).model {
            self.yoloModel = try? VNCoreMLModel(for: mlModel)
        } else {
            print("⚠️ YOLOモデルの読み込みに失敗しました。Xcodeのプロジェクトに.mlpackageを追加してください。")
        }
    }
    
    // 2. 古いスマホ（カメラ）への接続開始
    func startStream(url: String) {
        let config = URLSessionConfiguration.default
        session = URLSession(configuration: config, delegate: self, delegateQueue: nil)
        if let streamURL = URL(string: url) {
            dataTask = session?.dataTask(with: streamURL)
            dataTask?.resume()
        }
    }
    
    // 【ここにURLSessionDelegateを用いたMJPEG映像の受信処理が入ります】
    // （毎秒届くJPGE画像を UIImage に変換し、以下の analyzeFrame() に渡します）
    
    // 3. AIによる車両検知
    func analyzeFrame(image: UIImage) {
        guard let cgImage = image.cgImage, let model = yoloModel else { return }
        
        let request = VNCoreMLRequest(model: model) { [weak self] request, error in
            self?.processYOLOResult(request: request, image: cgImage)
        }
        
        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        try? handler.perform([request])
    }
    
    // 4. 検知結果の処理と、パトランプ（赤色）の検索
    private func processYOLOResult(request: VNRequest, image: CGImage) {
        guard let results = request.results as? [VNRecognizedObjectObservation] else { return }
        var foundPolice = false
        
        for observation in results {
            let label = observation.labels.first?.identifier ?? ""
            
            // 車、バイクを検知した場合
            if ["car", "motorcycle", "bus", "truck"].contains(label) && observation.confidence > 0.3 {
                
                // Y座標を上に25%拡張（屋根のパトランプ対策）
                let box = observation.boundingBox
                let expandedBox = CGRect(x: box.origin.x, y: max(0, box.origin.y - box.height * 0.25), width: box.width, height: box.height * 1.25)
                
                // Swiftで画像内の「赤色の面積」を計算する（Pythonでの処理の代替）
                if checkRedSirenRatio(in: expandedBox, cgImage: image) {
                    foundPolice = true
                }
            }
        }
        
        // 5. アラートの判定（Pythonと同じく一瞬見失っても維持するロジック）
        DispatchQueue.main.async {
            if foundPolice {
                self.emergencyVehicleFrames += 1
            } else {
                self.emergencyVehicleFrames = max(0, self.emergencyVehicleFrames - 1)
            }
            
            if self.emergencyVehicleFrames > 1 {
                self.triggerAlarm()
            } else {
                self.alertMessage = "安全（異常なし）"
                self.isAlerting = false
            }
        }
    }
    
    // 6. 赤色の割合チェックロジック（遠くの小さな赤い光 ＆ 赤い車の除外）
    private func checkRedSirenRatio(in rect: CGRect, cgImage: CGImage) -> Bool {
        // ※ 実際のSwiftではここに CoreImage や CGContext を使ったピクセルループ処理を記載し、
        // 「赤の割合が 0.05% 以上 12% 以下」であれば True を返すようにします。
        return true // テスト用
    }
    
    // 7. iPhoneでの激しい警告発砲
    private func triggerAlarm() {
        let now = Date()
        if now.timeIntervalSince(lastAlertTime) > 5 { // 5秒間隔
            self.alertMessage = "🚨警告：後方にパトカー検知！！"
            self.isAlerting = true
            
            // iPhone標準の通知音（テケテン！）を鳴らす
            AudioServicesPlaySystemSound(1005) 
            // バイブレーションを激しく鳴らす
            AudioServicesPlaySystemSound(kSystemSoundID_Vibrate)
            
            self.lastAlertTime = now
        }
    }
}

// MARK: - アプリの画面UI（SwiftUI）
struct ContentView: View {
    @StateObject var analyzer = DashcamAnalyzer()
    
    var body: some View {
        VStack(spacing: 30) {
            Text("AI ドライブレコーダー")
                .font(.largeTitle)
                .bold()
            
            // 警告表示エリア
            Text(analyzer.alertMessage)
                .font(.title)
                .foregroundColor(analyzer.isAlerting ? .white : .green)
                .padding()
                .frame(maxWidth: .infinity)
                .background(analyzer.isAlerting ? Color.red : Color.black)
                .cornerRadius(15)
            
            Button(action: {
                // 古いスマホのIPWebcamのURLを指定して監視スタート
                analyzer.startStream(url: "http://192.168.0.10:8080/video")
            }) {
                Text("カメラ接続＆監視スタート")
                    .font(.headline)
                    .padding()
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(10)
            }
        }
        .padding()
    }
}
