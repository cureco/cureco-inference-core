using System.IO;
using System.Net;
using System.Net.Http;
using System.Net.Sockets;
using System.Text.Json;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using Cureco.InferenceApp;

static void Check(bool value, string message) { if (!value) throw new Exception(message); }
if (args.Length is < 1 or > 2) throw new ArgumentException("Pass generated ONNX fixtures and optionally the published app executable");
var bitmap = BitmapSource.Create(2, 2, 96, 96, PixelFormats.Rgb24, null,
    new byte[] { 255,0,0, 255,0,0, 255,0,0, 255,0,0 }, 6);
var encoder = new PngBitmapEncoder(); encoder.Frames.Add(BitmapFrame.Create(bitmap));
using var stream = new MemoryStream(); encoder.Save(stream);
byte[] png = stream.ToArray();
var image = InferenceSession.Decode(png);
string exportDirectory = Path.Combine(Path.GetFullPath(args[0]), "exports-" + Guid.NewGuid().ToString("N"));
Directory.CreateDirectory(exportDirectory);
foreach (string task in new[] { "classification", "detection", "segmentation" })
{
    using var session = new InferenceSession(Path.Combine(args[0], task + ".onnx"));
    var output = session.Infer(image);
    Check(output.Data.GetProperty("task").GetString() == task, "Wrong task");
    if (task == "segmentation") Check(output.Mask?.Length == 4, "Wrong mask size");
    var rendered = ResultRenderer.Render(image, output);
    Check(rendered.PixelWidth == 2, "Rendering failed");
    string jsonPath = Path.Combine(exportDirectory, task + ".json");
    string pngPath = Path.Combine(exportDirectory, task + ".png");
    await ResultExporter.SaveJsonAsync(jsonPath, output);
    await ResultExporter.SavePngAsync(pngPath, rendered);
    using (var saved = JsonDocument.Parse(await File.ReadAllTextAsync(jsonPath)))
    {
        Check(saved.RootElement.GetProperty("result").GetProperty("task").GetString() == task, "Saved task mismatch");
        if (task == "segmentation")
            Check(saved.RootElement.GetProperty("mask").EnumerateArray().Select(value => value.GetInt32()).SequenceEqual(output.Mask!), "Saved mask mismatch");
    }
    Check(InferenceSession.Decode(await File.ReadAllBytesAsync(pngPath)).Width == 2, "Saved PNG cannot be reopened");
    foreach (var path in new[] { jsonPath, pngPath })
    {
        byte[] original = await File.ReadAllBytesAsync(path);
        using (var locked = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.None))
        {
            try
            {
                if (path == jsonPath) await ResultExporter.SaveJsonAsync(path, output);
                else await ResultExporter.SavePngAsync(path, rendered);
                throw new Exception("Overwrote locked output");
            }
            catch (Exception error) when (error is IOException or UnauthorizedAccessException) { }
        }
        Check(original.SequenceEqual(await File.ReadAllBytesAsync(path)), "Failed export modified existing output");
    }
    Check(!Directory.EnumerateFiles(exportDirectory, "*.tmp").Any(), "Temporary export left behind");
}
Console.WriteLine("PASS: JSON/PNG roundtrip for three tasks and existing-file protection on save failure");
try { using var rejected = new InferenceSession(Path.Combine(args[0], "parameters.onnx")); throw new Exception("Accepted auxiliary input"); }
catch (NotSupportedException) { }
try { InferenceSession.Decode(new byte[0]); throw new Exception("Accepted empty image"); }
catch (InvalidDataException) { }
string token = new('A', 64);
Check(new Settings().WithToken(token).Token() == token, "Protected token roundtrip failed");
using var listener = new TcpListener(IPAddress.Loopback, 0);
listener.Start(); int port = ((IPEndPoint)listener.LocalEndpoint).Port; listener.Stop();
using var active = new InferenceSession(Path.Combine(args[0], "classification.onnx"));
await using var server = await ApiServer.StartAsync(active, port, token);
using var client = new HttpClient { BaseAddress = new Uri($"http://127.0.0.1:{port}"), Timeout = TimeSpan.FromSeconds(10) };
using (var unauthorized = await client.GetAsync("/health")) Check(unauthorized.StatusCode == HttpStatusCode.Unauthorized, "Authentication missing");
client.DefaultRequestHeaders.Authorization = new("Bearer", token);
using (var health = await client.GetAsync("/health")) Check(health.IsSuccessStatusCode, "Health failed");
using (var invalid = await client.PostAsync("/infer", new StringContent("invalid"))) Check((int)invalid.StatusCode == 415, "Content type not checked");
using (var content = new ByteArrayContent(png))
{
    content.Headers.ContentType = new("image/png");
    using var response = await client.PostAsync("/infer", content);
    Check(response.IsSuccessStatusCode, "API inference failed");
    using var json = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
    Check(json.RootElement.GetProperty("result").GetProperty("task").GetString() == "classification", "API result mismatch");
}
client.DefaultRequestHeaders.Add("Origin", "https://example.com");
using (var origin = await client.GetAsync("/health")) Check(origin.StatusCode == HttpStatusCode.Forbidden, "Origin not rejected");
client.DefaultRequestHeaders.Remove("Origin");
client.DefaultRequestHeaders.Host = "example.com";
using (var host = await client.GetAsync("/health")) Check(host.StatusCode == HttpStatusCode.Forbidden, "Host not rejected");
Console.WriteLine("PASS: three tasks, renderer, auxiliary-input rejection, protected settings, API authentication/inference/origin/host");
Exception? windowError = null;
var ui = new Thread(() =>
{
    try
    {
        var application = new System.Windows.Application();
        var window = new MainWindow(new Settings());
        window.Loaded += (_, _) => window.Dispatcher.BeginInvoke(new Action(window.Close));
        application.Run(window);
    }
    catch (Exception error) { windowError = error; }
});
ui.SetApartmentState(ApartmentState.STA);
ui.Start();
if (!ui.Join(TimeSpan.FromSeconds(10))) throw new Exception("GUI shutdown timed out");
if (windowError != null) throw new Exception("GUI lifecycle failed", windowError);
Console.WriteLine("PASS: GUI startup and asynchronous shutdown");
if (args.Length == 2)
{
    string settingDirectory = Path.Combine(Path.GetFullPath(args[0]), "settings-" + Guid.NewGuid().ToString("N"));
    string? previous = Environment.GetEnvironmentVariable("CURECO_INFERENCE_SETTINGS_DIR");
    listener.Start(); int headlessPort = ((IPEndPoint)listener.LocalEndpoint).Port; listener.Stop();
    try
    {
        Environment.SetEnvironmentVariable("CURECO_INFERENCE_SETTINGS_DIR", settingDirectory);
        var saved = new Settings { ModelPath = Path.GetFullPath(Path.Combine(args[0], "classification.onnx")), Port = headlessPort }.WithToken(token);
        saved.Save();
        Check(Settings.Load().ModelPath == saved.ModelPath, "Settings roundtrip failed");
        var start = new System.Diagnostics.ProcessStartInfo(Path.GetFullPath(args[1]), "--serve")
        { UseShellExecute = false, CreateNoWindow = true, WindowStyle = System.Diagnostics.ProcessWindowStyle.Hidden };
        using (var process = System.Diagnostics.Process.Start(start)!)
        {
            try
            {
                using var probe = new HttpClient { BaseAddress = new Uri($"http://127.0.0.1:{headlessPort}"), Timeout = TimeSpan.FromSeconds(1) };
                probe.DefaultRequestHeaders.Authorization = new("Bearer", token);
                bool ready = false;
                for (int i = 0; i < 50 && !process.HasExited; i++)
                {
                    try { using var reply = await probe.GetAsync("/health"); ready = reply.IsSuccessStatusCode; }
                    catch (HttpRequestException) { }
                    catch (TaskCanceledException) { }
                    if (ready) break;
                    await Task.Delay(100);
                }
                Check(ready && process.MainWindowHandle == IntPtr.Zero, "Headless automatic startup failed");
            }
            finally { if (!process.HasExited) process.Kill(); await process.WaitForExitAsync(); }
        }
        (saved with { ModelPath = Path.Combine(settingDirectory, "missing.onnx") }).Save();
        using var failed = System.Diagnostics.Process.Start(start)!;
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(15));
        try { await failed.WaitForExitAsync(timeout.Token); }
        finally { if (!failed.HasExited) failed.Kill(); }
        Check(failed.ExitCode == 1 && File.Exists(Path.Combine(settingDirectory, "startup-error.log")), "Startup failure not reported");
        Console.WriteLine("PASS: saved settings, published headless API autostart, invalid model exit code");
    }
    finally { Environment.SetEnvironmentVariable("CURECO_INFERENCE_SETTINGS_DIR", previous); }
}
