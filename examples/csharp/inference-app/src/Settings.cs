using System.IO;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

namespace Cureco.InferenceApp;
public sealed record Settings
{
    public string ModelPath { get; init; } = "";
    public int Port { get; init; } = 8765;
    public bool StartServer { get; init; }
    public bool StartMinimized { get; init; }
    public string ProtectedToken { get; init; } = "";
    public static string DirectoryPath => Environment.GetEnvironmentVariable("CURECO_INFERENCE_SETTINGS_DIR") is { Length: > 0 } custom
        ? Path.GetFullPath(custom)
        : Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Cureco", "InferenceApp");
    static string FilePath => Path.Combine(DirectoryPath, "settings.json");
    public static Settings Load() => File.Exists(FilePath)
        ? JsonSerializer.Deserialize<Settings>(File.ReadAllText(FilePath)) ?? throw new InvalidDataException("Invalid settings")
        : new Settings();
    public Settings WithToken(string token) => this with { ProtectedToken = Convert.ToBase64String(ProtectedData.Protect(Encoding.UTF8.GetBytes(token), null, DataProtectionScope.CurrentUser)) };
    public string Token() => ProtectedToken.Length == 0 ? "" : Encoding.UTF8.GetString(ProtectedData.Unprotect(Convert.FromBase64String(ProtectedToken), null, DataProtectionScope.CurrentUser));
    public Settings EnsureToken() => ProtectedToken.Length == 0 ? RegenerateToken() : this;
    public Settings RegenerateToken()
    {
        var updated = WithToken(Convert.ToHexString(RandomNumberGenerator.GetBytes(32)));
        updated.Save();
        return updated;
    }
    public void Save()
    {
        if (Port < 1024 || Port > 65535) throw new ArgumentException("Port must be 1024–65535");
        Directory.CreateDirectory(Path.GetDirectoryName(FilePath)!);
        string temp = FilePath + "." + Guid.NewGuid().ToString("N") + ".tmp";
        try { File.WriteAllText(temp, JsonSerializer.Serialize(this, new JsonSerializerOptions { WriteIndented = true })); File.Move(temp, FilePath, true); }
        finally { if (File.Exists(temp)) File.Delete(temp); }
    }
}
