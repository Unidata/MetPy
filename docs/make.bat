@ECHO OFF

pushd %~dp0

REM Command file for Sphinx documentation

if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=sphinx-build
)
if "%SPHINXAUTOGEN%" == "" (
	set SPHINXAUTOGEN=sphinx-autogen
)
set SOURCEDIR=.
set BUILDDIR=build
set SPHINXPROJ=MetPy

if "%1" == "" goto help

if "%1" == "clean" (
	for /d %%i in (%BUILDDIR%\*) do rmdir /q /s %%i
	rmdir /q /s %SOURCEDIR%\examples\ %SOURCEDIR%\tutorials\ %SOURCEDIR%\api\generated\
	goto end
)

if "%1" == "overridecheck" (
	python override_check.py
	goto end
)

%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
	echo.
	echo.The 'sphinx-build' command was not found. Make sure you have Sphinx
	echo.installed, then set the SPHINXBUILD environment variable to point
	echo.to the full path of the 'sphinx-build' executable. Alternatively you
	echo.may add the Sphinx directory to PATH.
	echo.
	echo.If you don't have Sphinx installed, grab it from
	echo.http://sphinx-doc.org/
	exit /b 1
)

echo.Running sphinx-autogen
for %%i in (%SOURCEDIR%\api\*.rst) do %SPHINXAUTOGEN% -i -t %SOURCEDIR%\_templates -o %SOURCEDIR%\api\generated %%i
%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS%
goto end

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS%

:end
popd
