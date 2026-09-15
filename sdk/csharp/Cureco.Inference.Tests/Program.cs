using Cureco.Inference;
if (args.Length != 1) throw new ArgumentException("Pass a classification fixture");
var engine = new Engine(args[0]);
var image = new byte[] { 255,0,0 };
if (engine.Infer(image, 1, 1, 3).Data.GetProperty("task").GetString() != "classification") throw new Exception("Invalid result");
engine.Dispose(); engine.Dispose();
try { _ = engine.ModelInfo; throw new Exception("Disposed metadata access accepted"); } catch (ObjectDisposedException) { }
try { engine.Infer(image, 1, 1, 3); throw new Exception("Disposed inference accepted"); } catch (ObjectDisposedException) { }
Console.WriteLine("PASS: packaged binding inference and disposal");
