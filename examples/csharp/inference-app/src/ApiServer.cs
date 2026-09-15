using System.Net;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;

namespace Cureco.InferenceApp;
public sealed class ApiServer : IAsyncDisposable
{
    readonly WebApplication app;
    readonly SemaphoreSlim gate = new(1, 1);
    ApiServer(WebApplication app) => this.app = app;
    public static async Task<ApiServer> StartAsync(InferenceSession session, int port, string token, CancellationToken cancel = default)
    {
        if (port < 1024 || port > 65535 || token.Length < 32) throw new ArgumentException("Invalid port or API token");
        var builder = WebApplication.CreateSlimBuilder();
        builder.Logging.ClearProviders(); // Never write request bodies or bearer tokens to logs.
        builder.WebHost.ConfigureKestrel(options =>
        {
            options.Listen(IPAddress.Loopback, port);
            options.Limits.MaxRequestBodySize = InferenceSession.MaxBytes;
            options.Limits.RequestHeadersTimeout = TimeSpan.FromSeconds(10);
            options.Limits.MaxConcurrentConnections = 16;
        });
        var app = builder.Build();
        var server = new ApiServer(app);
        byte[] expected = SHA256.HashData(Encoding.UTF8.GetBytes("Bearer " + token));
        app.Use(async (context, next) =>
        {
            if (context.Request.Host.Host != "127.0.0.1" || context.Request.Headers.ContainsKey("Origin"))
            { context.Response.StatusCode = 403; return; }
            byte[] actual = SHA256.HashData(Encoding.UTF8.GetBytes(context.Request.Headers.Authorization.ToString()));
            if (!CryptographicOperations.FixedTimeEquals(expected, actual))
            { context.Response.StatusCode = 401; return; }
            await next(context);
        });
        app.MapGet("/health", () => Results.Ok(new { status = "ready" }));
        app.MapPost("/infer", async (HttpContext context) =>
        {
            if (!await server.gate.WaitAsync(0, context.RequestAborted)) return Results.StatusCode(429);
            try
            {
                if (context.Request.ContentType?.Split(';')[0] is not ("image/png" or "image/jpeg" or "image/bmp"))
                    return Results.StatusCode(415);
                using var data = new MemoryStream();
                byte[] buffer = new byte[81920];
                int count;
                while ((count = await context.Request.Body.ReadAsync(buffer, context.RequestAborted)) > 0)
                {
                    if (data.Length + count > InferenceSession.MaxBytes) return Results.StatusCode(413);
                    data.Write(buffer, 0, count);
                }
                var result = await Task.Run(() => session.Infer(InferenceSession.Decode(data.ToArray())), context.RequestAborted);
                return Results.Json(new { result = result.Data, mask = result.Mask });
            }
            catch (OperationCanceledException) { return Results.StatusCode(499); }
            catch (BadHttpRequestException error) { return Results.StatusCode(error.StatusCode); }
            catch { return Results.BadRequest(new { error = "Image could not be processed." }); }
            finally { server.gate.Release(); }
        });
        try { await app.StartAsync(cancel); return server; }
        catch { await server.DisposeAsync(); throw; }
    }
    public async ValueTask DisposeAsync()
    {
        await app.StopAsync();
        await app.DisposeAsync();
        gate.Dispose();
    }
}
