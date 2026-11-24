param(
    [string]$PythonVersion = '3.12.15'
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$decoderRoot = Join-Path $projectRoot '.decoder-venv'
$decoderPython = Join-Path $decoderRoot 'Scripts\python.exe'
$requirements = Join-Path $projectRoot 'configs\requirements-decoder.txt'
$previousPythonDir = $env:UV_PYTHON_INSTALL_DIR
$previousCacheDir = $env:UV_CACHE_DIR

function Invoke-Uv {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$UvArguments)
    & uv @UvArguments
    if ($LASTEXITCODE -ne 0) {
        throw "uv failed with exit code $LASTEXITCODE. Source builds are not part of this setup."
    }
}

try {
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        throw 'uv is required. Install uv before running this script.'
    }
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $projectRoot 'assets\local\python'
    $env:UV_CACHE_DIR = Join-Path $projectRoot 'assets\local\uv-cache'

    if (-not (Test-Path -LiteralPath $decoderPython)) {
        $pythonPath = & uv python find --no-python-downloads $PythonVersion 2>$null
        if ($LASTEXITCODE -ne 0) {
            Invoke-Uv @('python', 'install', $PythonVersion, '--no-bin', '--no-registry')
            $pythonPath = & uv python find --no-python-downloads --managed-python $PythonVersion
            if ($LASTEXITCODE -ne 0) { throw 'Could not locate the project Python installation.' }
        }
        Invoke-Uv @('venv', '--no-project', '--no-python-downloads', '--python', $pythonPath.Trim(), $decoderRoot)
    }

    & $decoderPython -c "import struct,sys; assert sys.version_info[:2] == (3,12) and struct.calcsize('P') == 8, 'Decoder requires CPython 3.12 x64'"
    if ($LASTEXITCODE -ne 0) { throw 'The existing decoder environment has an unsupported interpreter.' }

    # Wheel-only installation prevents an accidental source build on unsupported Python versions.
    Invoke-Uv @('pip', 'install', '--python', $decoderPython, '--only-binary', ':all:', '--requirements', $requirements, '--no-config')

    @'
import importlib.metadata as metadata
import json
from flashlight.lib.text.decoder import LexiconDecoder, Trie, ZeroLM

status = {name: metadata.version(name) for name in ('flashlight-text', 'numpy', 'psutil')}
status['core_decoder_import'] = 'passed'
try:
    from flashlight.lib.text.decoder.kenlm import KenLM
    status['kenlm_import'] = 'available'
except ImportError as error:
    status['kenlm_import'] = str(error).strip()
print(json.dumps(status, indent=2))
'@ | & $decoderPython -
    if ($LASTEXITCODE -ne 0) { throw 'The decoder import check failed.' }
} finally {
    $env:UV_PYTHON_INSTALL_DIR = $previousPythonDir
    $env:UV_CACHE_DIR = $previousCacheDir
}
