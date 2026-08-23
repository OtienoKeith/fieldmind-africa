param(
    [string]$RepoRoot = (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Drawing
Add-Type -AssemblyName System.Speech

$videoDir = Join-Path $RepoRoot "video"
$assetDir = Join-Path $videoDir "assets"
New-Item -ItemType Directory -Force -Path $assetDir | Out-Null

$width = 1280
$height = 720
$background = [System.Drawing.Color]::FromArgb(8, 19, 15)
$green = [System.Drawing.Color]::FromArgb(29, 104, 72)
$cream = [System.Drawing.Color]::FromArgb(255, 246, 214)
$white = [System.Drawing.Color]::FromArgb(245, 249, 247)
$muted = [System.Drawing.Color]::FromArgb(184, 205, 194)

function New-Canvas {
    $bitmap = New-Object System.Drawing.Bitmap($width, $height)
    $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
    $graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::HighQuality
    $graphics.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
    $graphics.Clear($background)
    return @($bitmap, $graphics)
}

function Save-TitleFrame {
    param([string]$Path, [string]$Kicker, [string]$Title, [string]$Subtitle)
    $canvas = New-Canvas
    $bitmap = $canvas[0]
    $graphics = $canvas[1]
    $graphics.FillRectangle((New-Object System.Drawing.SolidBrush($green)), 80, 90, 1120, 540)
    $kickerFont = New-Object System.Drawing.Font("Segoe UI", 19, [System.Drawing.FontStyle]::Bold)
    $titleFont = New-Object System.Drawing.Font("Segoe UI", 52, [System.Drawing.FontStyle]::Bold)
    $subtitleFont = New-Object System.Drawing.Font("Segoe UI", 25, [System.Drawing.FontStyle]::Regular)
    $graphics.DrawString($Kicker, $kickerFont, (New-Object System.Drawing.SolidBrush($cream)), 130, 145)
    $graphics.DrawString($Title, $titleFont, (New-Object System.Drawing.SolidBrush($white)), 125, 225)
    $subtitleRect = New-Object System.Drawing.RectangleF(132, 345, 1000, 150)
    $graphics.DrawString($Subtitle, $subtitleFont, (New-Object System.Drawing.SolidBrush($white)), $subtitleRect)
    $graphics.DrawString("ADTC 2026 | Free/open tooling | CPU llama.cpp", $kickerFont, (New-Object System.Drawing.SolidBrush($cream)), 132, 550)
    $bitmap.Save($Path, [System.Drawing.Imaging.ImageFormat]::Png)
    $graphics.Dispose()
    $bitmap.Dispose()
}

function Save-ScreenshotFrame {
    param([string]$Path, [string]$ImagePath, [string]$Caption)
    $canvas = New-Canvas
    $bitmap = $canvas[0]
    $graphics = $canvas[1]
    $source = [System.Drawing.Image]::FromFile($ImagePath)
    $maxWidth = 1180.0
    $maxHeight = 575.0
    $scale = [Math]::Min($maxWidth / $source.Width, $maxHeight / $source.Height)
    $drawWidth = [int]($source.Width * $scale)
    $drawHeight = [int]($source.Height * $scale)
    $drawX = [int](($width - $drawWidth) / 2)
    $drawY = [int](35 + (($maxHeight - $drawHeight) / 2))
    $graphics.DrawImage($source, $drawX, $drawY, $drawWidth, $drawHeight)
    $source.Dispose()
    $graphics.FillRectangle((New-Object System.Drawing.SolidBrush($green)), 0, 625, $width, 95)
    $captionFont = New-Object System.Drawing.Font("Segoe UI", 24, [System.Drawing.FontStyle]::Bold)
    $captionRect = New-Object System.Drawing.RectangleF(55, 644, 1170, 55)
    $format = New-Object System.Drawing.StringFormat
    $format.Alignment = [System.Drawing.StringAlignment]::Center
    $graphics.DrawString($Caption, $captionFont, (New-Object System.Drawing.SolidBrush($white)), $captionRect, $format)
    $bitmap.Save($Path, [System.Drawing.Imaging.ImageFormat]::Png)
    $graphics.Dispose()
    $bitmap.Dispose()
}

$frames = @(
    @{ Name = "01-title"; Text = "Twenty thousand naira can be a farmer's input budget. One wrong chemical purchase can consume it without treating the real problem." },
    @{ Name = "02-cloud"; Text = "FieldMind Africa is a free, text-only, offline-ready purchase decision assistant. The public Hugging Face demo runs without a paid inference API." },
    @{ Name = "03-cassava"; Text = "For cassava leaves with mosaic patches and curling, FieldMind does not merely list causes. It gives the purchase verdict: do not buy fungicide. Fungicide does not treat a virus, and the answer cites the IITA cassava guide." },
    @{ Name = "04-kiswahili"; Text = "FieldMind detects English and Kiswahili. For flooded young maize entered in Kiswahili, the complete answer stays in Kiswahili, says not to buy CAN yet, and puts drainage and field checks before fertilizer." },
    @{ Name = "05-repo"; Text = "The public repository contains the reproducible pipeline. It prepares eleven thousand two hundred and eighty open training records, validates zero question leakage, fine tunes Qwen three one point seven billion with free Colab Q LoRA, and exports CPU GGUF files for llama dot C P P." },
    @{ Name = "06-close"; Text = "FieldMind does not promise a laboratory diagnosis from one text message. It protects the decision before scarce money is spent. Before you buy the chemical, ask the laptop." }
)

Save-TitleFrame -Path (Join-Path $assetDir "01-title.png") -Kicker "NGN 20,000" -Title "FieldMind Africa" -Subtitle "Before you buy the chemical, ask the laptop."
Save-ScreenshotFrame -Path (Join-Path $assetDir "02-cloud.png") -ImagePath (Join-Path $RepoRoot "benchmarks/live-space-home.png") -Caption "FREE PUBLIC CLOUD DEMO | NO PAID API"
Save-ScreenshotFrame -Path (Join-Path $assetDir "03-cassava.png") -ImagePath (Join-Path $RepoRoot "benchmarks/video-cassava-verdict.png") -Caption "PURCHASE VERDICT: DO NOT BUY FUNGICIDE"
Save-ScreenshotFrame -Path (Join-Path $assetDir "04-kiswahili.png") -ImagePath (Join-Path $RepoRoot "benchmarks/video-kiswahili-verdict.png") -Caption "AUTOMATIC LANGUAGE ROUTING | COMPLETE KISWAHILI ANSWER"
Save-ScreenshotFrame -Path (Join-Path $assetDir "05-repo.png") -ImagePath (Join-Path $RepoRoot "benchmarks/public-github-repo.png") -Caption "PUBLIC REPRODUCIBLE PIPELINE | OPEN DATA | PASSING CHECKS"
Save-TitleFrame -Path (Join-Path $assetDir "06-close.png") -Kicker "SCIENCE | EDGE ENGINEERING | PRODUCT IMPACT" -Title "FieldMind Africa" -Subtitle "Live: huggingface.co/spaces/otieno28/fieldmind-africa"

$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.SelectVoice("Microsoft Zira Desktop")
$synth.Rate = 0
$synth.Volume = 100
foreach ($frame in $frames) {
    $wav = Join-Path $assetDir ($frame.Name + ".wav")
    $synth.SetOutputToWaveFile($wav)
    $synth.Speak($frame.Text)
    $synth.SetOutputToNull()
}
$synth.Dispose()

$clips = @()
foreach ($frame in $frames) {
    $png = Join-Path $assetDir ($frame.Name + ".png")
    $wav = Join-Path $assetDir ($frame.Name + ".wav")
    $clip = Join-Path $assetDir ($frame.Name + ".mp4")
    & ffmpeg -y -loglevel error -loop 1 -framerate 30 -i $png -i $wav -vf "fade=t=in:st=0:d=0.35,fade=t=out:st=9999:d=0.1" -c:v libx264 -preset medium -tune stillimage -pix_fmt yuv420p -c:a aac -b:a 160k -shortest -movflags +faststart $clip
    if ($LASTEXITCODE -ne 0) { throw "ffmpeg failed for $($frame.Name)" }
    $clips += $clip
}

$concatPath = Join-Path $assetDir "concat.txt"
$concatLines = $clips | ForEach-Object { "file '$($_.Replace("'", "''"))'" }
[System.IO.File]::WriteAllLines($concatPath, $concatLines)
$output = Join-Path $videoDir "fieldmind-africa-demo.mp4"
& ffmpeg -y -loglevel error -f concat -safe 0 -i $concatPath -c copy -movflags +faststart $output
if ($LASTEXITCODE -ne 0) { throw "ffmpeg concat failed" }

$duration = & ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 $output
Write-Host "Created $output ($duration seconds)"
