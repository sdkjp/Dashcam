import SwiftUI

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
                // 【注意】ここに古いスマホのIPWebcamのURLを指定します
                analyzer.startStream(url: "http://192.168.x.x:8080/video")
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
