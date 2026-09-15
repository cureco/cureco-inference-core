using System.IO;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using Cureco.Inference;

namespace Cureco.InferenceApp;
public sealed record DecodedImage(byte[] Pixels, int Width, int Height, BitmapSource Preview);
public sealed class InferenceSession : IDisposable
{
    public const int MaxPixels = 16 * 1024 * 1024;
    public const int MaxBytes = 20 * 1024 * 1024;
    readonly Engine engine;
    public InferenceSession(string path)
    {
        engine = new Engine(path);
        try
        {
            foreach (var input in engine.ModelInfo.GetProperty("inputs").EnumerateArray())
            {
                var shape = input.GetProperty("shape");
                if (shape.GetArrayLength() != 4 && (shape.GetArrayLength() != 2 || shape[1].GetInt64() != 0))
                    throw new NotSupportedException("This app accepts image-only models. Use the SDK for models requiring auxiliary inputs.");
            }
        }
        catch { engine.Dispose(); throw; }
    }
    public string Info => engine.ModelInfo.GetRawText();
    public static DecodedImage Decode(byte[] bytes)
    {
        if (bytes.Length == 0 || bytes.Length > MaxBytes) throw new InvalidDataException("Image must be 1 byte–20 MiB");
        using var stream = new MemoryStream(bytes, false);
        var frame = BitmapDecoder.Create(stream, BitmapCreateOptions.PreservePixelFormat, BitmapCacheOption.OnDemand).Frames[0];
        if ((long)frame.PixelWidth * frame.PixelHeight > MaxPixels) throw new InvalidDataException("Image exceeds 16 megapixels");
        var bitmap = new FormatConvertedBitmap(frame, PixelFormats.Bgr24, null, 0);
        bitmap.Freeze();
        byte[] pixels = new byte[checked(bitmap.PixelWidth * bitmap.PixelHeight * 3)];
        bitmap.CopyPixels(pixels, bitmap.PixelWidth * 3, 0);
        var preview = BitmapSource.Create(bitmap.PixelWidth, bitmap.PixelHeight, 96, 96, PixelFormats.Bgr24, null, pixels, bitmap.PixelWidth * 3);
        preview.Freeze();
        return new DecodedImage(pixels, bitmap.PixelWidth, bitmap.PixelHeight, preview);
    }
    public InferenceResult Infer(DecodedImage image) => engine.Infer(image.Pixels, image.Width, image.Height, image.Width * 3, Cureco.Inference.PixelFormat.BGR);
    public void Dispose() => engine.Dispose();
}
