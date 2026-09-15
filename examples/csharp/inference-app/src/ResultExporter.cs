using System.IO;
using System.Text.Json;
using System.Windows.Media.Imaging;
using Cureco.Inference;

namespace Cureco.InferenceApp;
public static class ResultExporter
{
    public static Task SaveJsonAsync(string path, InferenceResult result) =>
        WriteAtomicAsync(path, stream => JsonSerializer.SerializeAsync(stream,
            new { result = result.Data, mask = result.Mask },
            new JsonSerializerOptions { WriteIndented = true }));

    public static Task SavePngAsync(string path, BitmapSource image) =>
        WriteAtomicAsync(path, stream =>
        {
            var encoder = new PngBitmapEncoder();
            encoder.Frames.Add(BitmapFrame.Create(image));
            encoder.Save(stream);
            return Task.CompletedTask;
        });

    static async Task WriteAtomicAsync(string path, Func<Stream, Task> write)
    {
        string target = Path.GetFullPath(path);
        string temporary = Path.Combine(Path.GetDirectoryName(target)!, "." + Path.GetFileName(target) + "." + Guid.NewGuid().ToString("N") + ".tmp");
        try
        {
            await using (var file = new FileStream(temporary, FileMode.CreateNew, FileAccess.Write, FileShare.None, 65536, FileOptions.Asynchronous))
            {
                await write(file);
                await file.FlushAsync();
            }
            File.Move(temporary, target, true);
        }
        finally { if (File.Exists(temporary)) File.Delete(temporary); }
    }
}
