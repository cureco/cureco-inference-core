using System.IO;
using System.Windows;

namespace Cureco.InferenceApp;
public static class Program
{
    [STAThread]
    public static int Main(string[] args)
    {
        try
        {
            if (args.Length == 1 && args[0] == "--serve") return ServeAsync().GetAwaiter().GetResult();
            if (args.Length != 0) throw new ArgumentException("Usage: CurecoInference.exe [--serve]");
            var application = new Application();
            application.Run(new MainWindow(Settings.Load()));
            return 0;
        }
        catch (Exception error)
        {
            if (args.Contains("--serve"))
            {
                var folder = Settings.DirectoryPath;
                Directory.CreateDirectory(folder);
                File.WriteAllText(Path.Combine(folder, "startup-error.log"), DateTimeOffset.Now + "\n" + error.Message);
            }
            else MessageBox.Show(error.Message, "Cureco Inference", MessageBoxButton.OK, MessageBoxImage.Error);
            return 1;
        }
    }
    static async Task<int> ServeAsync()
    {
        var settings = Settings.Load();
        using var session = new InferenceSession(settings.ModelPath);
        using var stop = new CancellationTokenSource();
        Console.CancelKeyPress += (_, e) => { e.Cancel = true; stop.Cancel(); };
        await using var server = await ApiServer.StartAsync(session, settings.Port, settings.Token(), stop.Token);
        try { await Task.Delay(Timeout.Infinite, stop.Token); } catch (OperationCanceledException) { }
        return 0;
    }
}
