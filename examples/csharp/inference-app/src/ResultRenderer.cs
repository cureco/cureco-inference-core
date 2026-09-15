using System.Globalization;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using Cureco.Inference;

namespace Cureco.InferenceApp;
public static class ResultRenderer
{
    static Color ColorFor(int id) => Color.FromRgb((byte)(50 + (uint)id * 67 % 180), (byte)(50 + (uint)id * 113 % 180), (byte)(50 + (uint)id * 151 % 180));
    public static BitmapSource Render(DecodedImage image, InferenceResult result)
    {
        BitmapSource background = image.Preview;
        if (result.Mask is { } mask)
        {
            if (mask.Length != image.Width * image.Height) throw new InvalidOperationException("Unexpected mask dimensions");
            var pixels = (byte[])image.Pixels.Clone();
            for (int i = 0; i < mask.Length; i++)
            {
                var color = ColorFor(mask[i]);
                pixels[3*i] = (byte)((pixels[3*i] + color.B) / 2);
                pixels[3*i+1] = (byte)((pixels[3*i+1] + color.G) / 2);
                pixels[3*i+2] = (byte)((pixels[3*i+2] + color.R) / 2);
            }
            background = BitmapSource.Create(image.Width, image.Height, 96, 96, PixelFormats.Bgr24, null, pixels, image.Width * 3);
            background.Freeze();
        }
        var visual = new DrawingVisual();
        using (var dc = visual.RenderOpen())
        {
            dc.DrawImage(background, new Rect(0, 0, image.Width, image.Height));
            if (result.Data.TryGetProperty("detections", out var detections))
            {
                foreach (var detection in detections.EnumerateArray())
                {
                    var box = detection.GetProperty("box");
                    var brush = new SolidColorBrush(ColorFor(detection.GetProperty("class_id").GetInt32()));
                    double left = box[0].GetDouble(), top = box[1].GetDouble();
                    dc.DrawRectangle(null, new Pen(brush, Math.Max(2, image.Width / 400.0)),
                        new Rect(left, top, Math.Max(0, box[2].GetDouble()-left), Math.Max(0, box[3].GetDouble()-top)));
                    var text = new FormattedText(detection.GetProperty("label").GetString() + " " + detection.GetProperty("confidence").GetDouble().ToString("P0"),
                        CultureInfo.InvariantCulture, FlowDirection.LeftToRight, new Typeface("Segoe UI"), Math.Max(12, image.Width/60.0), Brushes.White, 1);
                    dc.DrawRectangle(Brushes.Black, null, new Rect(left, top, text.Width+6, text.Height+2));
                    dc.DrawText(text, new Point(left+3, top+1));
                }
            }
        }
        var output = new RenderTargetBitmap(image.Width, image.Height, 96, 96, PixelFormats.Pbgra32);
        output.Render(visual); output.Freeze(); return output;
    }
}
