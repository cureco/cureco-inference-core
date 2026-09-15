using System.Runtime.InteropServices;
using System.Text.Json;
using Microsoft.Win32.SafeHandles;

namespace Cureco.Inference;

internal static class Native
{
    const string Library = "cureco_inference";
    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)] internal static extern uint ci_abi_version();
    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr ci_last_error();
    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)] internal static extern int ci_create(
        [MarshalAs(UnmanagedType.LPUTF8Str)] string path, [MarshalAs(UnmanagedType.LPUTF8Str)] string options, out IntPtr handle);
    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)] internal static extern void ci_destroy(IntPtr handle);
    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr ci_model_info(EngineHandle handle);
    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)] internal static extern unsafe int ci_run(
        EngineHandle handle, byte* pixels, nuint bytes, int width, int height, nuint stride, int format,
        float* parameters, nuint count, float threshold, out IntPtr result);
    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr ci_result_json(ResultHandle handle);
    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr ci_result_mask(ResultHandle handle, out nuint count);
    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)] internal static extern void ci_result_destroy(IntPtr handle);
    internal static void Check(int status) { if (status != 0) throw new InvalidOperationException(Marshal.PtrToStringUTF8(ci_last_error())); }
}
internal sealed class EngineHandle : SafeHandleZeroOrMinusOneIsInvalid
{
    internal EngineHandle(IntPtr value) : base(true) => SetHandle(value);
    protected override bool ReleaseHandle() { Native.ci_destroy(handle); return true; }
}
internal sealed class ResultHandle : SafeHandleZeroOrMinusOneIsInvalid
{
    internal ResultHandle(IntPtr value) : base(true) => SetHandle(value);
    protected override bool ReleaseHandle() { Native.ci_result_destroy(handle); return true; }
}
public enum PixelFormat { RGB = 0, BGR = 1, Gray = 2 }
public sealed record InferenceResult(JsonElement Data, int[]? Mask);
public sealed class Engine : IDisposable
{
    private readonly EngineHandle handle;
    private readonly object gate = new();
    public Engine(string modelPath, string optionsJson = "{}")
    {
        if (Native.ci_abi_version() != 1) throw new NotSupportedException("Unsupported native ABI");
        Native.Check(Native.ci_create(modelPath, optionsJson, out var value));
        handle = new EngineHandle(value);
    }
    public JsonElement ModelInfo { get { lock (gate) { ObjectDisposedException.ThrowIf(handle.IsClosed, this); return JsonSerializer.Deserialize<JsonElement>(Marshal.PtrToStringUTF8(Native.ci_model_info(handle))!); } } }
    public unsafe InferenceResult Infer(byte[] image, int width, int height, int stride,
                                       PixelFormat format = PixelFormat.RGB, float[]? parameters = null, float threshold = 0.25f)
    {
        lock (gate)
        {
        ObjectDisposedException.ThrowIf(handle.IsClosed, this);
        ArgumentNullException.ThrowIfNull(image);
        if (stride < 0) throw new ArgumentOutOfRangeException(nameof(stride));
        parameters ??= Array.Empty<float>();
        IntPtr value;
        fixed (byte* pixels = image)
        fixed (float* values = parameters)
            Native.Check(Native.ci_run(handle, pixels, (nuint)image.Length, width, height, (nuint)stride,
                                      (int)format, values, (nuint)parameters.Length, threshold, out value));
        using var result = new ResultHandle(value);
        var data = JsonSerializer.Deserialize<JsonElement>(Marshal.PtrToStringUTF8(Native.ci_result_json(result))!);
        var ptr = Native.ci_result_mask(result, out var count);
        int[]? mask = null;
        if (count > 0) { mask = new int[checked((int)count)]; Marshal.Copy(ptr, mask, 0, mask.Length); }
        return new InferenceResult(data, mask);
        }
    }
    public void Dispose() { lock (gate) handle.Dispose(); }
}
