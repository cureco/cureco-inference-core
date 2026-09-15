using System.ComponentModel;
using System.IO;
using System.Text.Json;
using System.Windows;
using System.Windows.Media.Imaging;
using Cureco.Inference;
using Microsoft.Win32;

namespace Cureco.InferenceApp;
public partial class MainWindow : Window
{
    Settings settings;
    string token;
    InferenceSession? session;
    ApiServer? server;
    DecodedImage? image;
    InferenceResult? result;
    BitmapSource? display;
    string modelPath = "";
    bool busy, canClose;

    public MainWindow(Settings settings)
    {
        InitializeComponent();
        this.settings = settings;
        token = settings.Token();
        Port.Text = settings.Port.ToString();
        AutoServer.IsChecked = settings.StartServer;
        Minimized.IsChecked = settings.StartMinimized;
        if (settings.StartMinimized) WindowState = WindowState.Minimized;
        Loaded += async (_, _) => await Work(async () =>
        {
            if (settings.ModelPath.Length > 0) await LoadModel(settings.ModelPath);
            if (settings.StartServer) await StartServer();
        });
        Closing += OnClosing;
    }
    async Task Work(Func<Task> action)
    {
        if (busy) return;
        busy = true; Actions.IsEnabled = ServerControls.IsEnabled = false;
        try { await action(); }
        catch (Exception error) { Status.Text = error.Message; }
        finally { busy = false; Actions.IsEnabled = ServerControls.IsEnabled = true; }
    }
    async Task LoadModel(string path)
    {
        if (server != null) throw new InvalidOperationException("モデルを変更する前にAPIを停止してください。");
        Status.Text = "モデルを読み込み中…";
        var loaded = await Task.Run(() => new InferenceSession(path));
        session?.Dispose(); session = loaded; modelPath = path;
        ModelName.Text = Path.GetFileName(path);
        result = null; ResultText.Text = ""; display = image?.Preview; Preview.Source = display;
        Status.Text = "モデルを読み込みました。画像を選択して推論できます。";
    }
    async void OpenModel(object sender, RoutedEventArgs e)
    {
        var dialog = new OpenFileDialog { Filter = "ONNX model|*.onnx" };
        if (dialog.ShowDialog(this) == true) await Work(() => LoadModel(dialog.FileName));
    }
    async void OpenImage(object sender, RoutedEventArgs e)
    {
        var dialog = new OpenFileDialog { Filter = "Images|*.png;*.jpg;*.jpeg;*.bmp" };
        if (dialog.ShowDialog(this) != true) return;
        await Work(async () =>
        {
            if (new FileInfo(dialog.FileName).Length > InferenceSession.MaxBytes) throw new InvalidDataException("画像は20 MiB以下にしてください。");
            var decoded = await Task.Run(() => InferenceSession.Decode(File.ReadAllBytes(dialog.FileName)));
            image = decoded; display = decoded.Preview; Preview.Source = display; EmptyHint.Visibility = Visibility.Collapsed;
            ImageName.Text = Path.GetFileName(dialog.FileName) + $"  {image.Width} × {image.Height}";
            result = null; ResultText.Text = ""; Status.Text = "画像を読み込みました。";
        });
    }
    async void RunInference(object sender, RoutedEventArgs e) => await Work(async () =>
    {
        if (session == null || image == null) throw new InvalidOperationException("モデルと画像を選択してください。");
        Status.Text = "推論中…";
        var timer = System.Diagnostics.Stopwatch.StartNew();
        var output = await Task.Run(() => session.Infer(image));
        var rendered = await Task.Run(() => ResultRenderer.Render(image, output));
        result = output; display = rendered; Preview.Source = rendered;
        ResultText.Text = JsonSerializer.Serialize(output.Data, new JsonSerializerOptions { WriteIndented = true });
        Status.Text = $"推論完了  /  {timer.ElapsedMilliseconds:N0} ms";
    });
    Settings CurrentSettings()
    {
        if (!int.TryParse(Port.Text, out int port) || port < 1024 || port > 65535) throw new ArgumentException("ポートは1024～65535で指定してください。");
        return settings with { ModelPath = modelPath, Port = port, StartServer = AutoServer.IsChecked == true, StartMinimized = Minimized.IsChecked == true };
    }
    async void SaveSettings(object sender, RoutedEventArgs e) => await Work(() =>
    {
        if (session == null) throw new InvalidOperationException("起動時に使用するモデルを選択してください。");
        settings = CurrentSettings().WithToken(token); settings.Save();
        Status.Text = "起動設定を保存しました。OSへの自動起動登録は行いません。";
        return Task.CompletedTask;
    });
    async Task StartServer()
    {
        if (session == null) throw new InvalidOperationException("モデルを選択してください。");
        var current = CurrentSettings();
        server = await ApiServer.StartAsync(session, current.Port, token);
        ServerButton.Content = "APIを停止"; ServerStatus.Text = $"稼働中  http://127.0.0.1:{current.Port}";
        Port.IsEnabled = false;
        Status.Text = "APIを開始しました。";
    }
    async void ToggleServer(object sender, RoutedEventArgs e) => await Work(async () =>
    {
        if (server == null) await StartServer();
        else
        {
            await server.DisposeAsync(); server = null; Port.IsEnabled = true;
            ServerButton.Content = "APIを開始"; ServerStatus.Text = "停止中"; Status.Text = "APIを停止しました。";
        }
    });
    async void CopyToken(object sender, RoutedEventArgs e) => await Work(() =>
    {
        Clipboard.SetText(token); Status.Text = "APIトークンをコピーしました。共有・公開しないでください。";
        return Task.CompletedTask;
    });
    async void RegenerateToken(object sender, RoutedEventArgs e) => await Work(() =>
    {
        if (server != null) throw new InvalidOperationException("トークンを再発行する前にAPIを停止してください。");
        if (MessageBox.Show(this, "以前のトークンは使えなくなります。APIを呼び出すプログラムの設定も更新してください。再発行しますか？",
            "APIトークンの再発行", MessageBoxButton.YesNo, MessageBoxImage.Warning, MessageBoxResult.No) != MessageBoxResult.Yes)
            return Task.CompletedTask;
        settings = settings.RegenerateToken();
        token = settings.Token();
        Status.Text = "APIトークンを再発行して保存しました。";
        return Task.CompletedTask;
    });
    async void SaveJson(object sender, RoutedEventArgs e) => await Work(async () =>
    {
        if (result == null) throw new InvalidOperationException("先に推論を実行してください。");
        var dialog = new SaveFileDialog { Filter = "JSON|*.json", FileName = "result.json" };
        if (dialog.ShowDialog(this) == true)
        {
            await ResultExporter.SaveJsonAsync(dialog.FileName, result);
            Status.Text = "JSONを保存しました。";
        }
    });
    async void SavePng(object sender, RoutedEventArgs e) => await Work(async () =>
    {
        if (display == null || result == null) throw new InvalidOperationException("先に推論を実行してください。");
        var dialog = new SaveFileDialog { Filter = "PNG|*.png", FileName = "result.png" };
        if (dialog.ShowDialog(this) == true)
        {
            await ResultExporter.SavePngAsync(dialog.FileName, display);
            Status.Text = "表示画像をPNG保存しました。";
        }
    });
    async void OnClosing(object? sender, CancelEventArgs e)
    {
        if (canClose) return;
        e.Cancel = true;
        if (busy) { Status.Text = "処理完了後に閉じてください。"; return; }
        await Work(async () =>
        {
            if (server != null) { await server.DisposeAsync(); server = null; }
            session?.Dispose(); session = null;
            canClose = true;
        });
        if (canClose) _ = Dispatcher.BeginInvoke(new Action(Close));
    }
}
