using Cureco.Inference;
try
{
    if (args.Length != 4) throw new ArgumentException("Usage: infer model.onnx image.rgb width height");
    int width = int.Parse(args[2]), height = int.Parse(args[3]);
    if (width < 1 || height < 1 || width > 16384 || height > 16384 || (long)width * height > 16777216)
        throw new ArgumentException("Invalid image dimensions");
    byte[] pixels = File.ReadAllBytes(args[1]);
    if (pixels.Length != checked(width * height * 3)) throw new ArgumentException("Invalid RGB byte count");
    using var engine = new Engine(args[0]);
    var result = engine.Infer(pixels, width, height, width * 3);
    Console.WriteLine(result.Data.GetRawText());
    return 0;
}
catch (Exception error) { Console.Error.WriteLine(error.Message); return 1; }
