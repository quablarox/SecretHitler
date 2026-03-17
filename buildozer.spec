[app]

# Title of the application
title = Secret Hitler

# Package name — must be unique on the Play Store
package.name = secrethitler

# Package domain (reverse-DNS convention)
package.domain = com.github.quablarox

# Source code dir
source.dir = .

# Python files to include (all .py files in the project)
source.include_exts = py,png,jpg,kv,atlas

# Application version
version = 0.1.0

# Main entry-point
entrypoint = main.py

# Requirements
# Kivy and its dependencies; python3 is implicit
requirements = python3,kivy

# Android orientation
orientation = portrait

# Android API levels
android.api = 33
android.minapi = 26
android.ndk = 25b

# Permissions
android.permissions = INTERNET

# Arch — build for the two most-common Android ABIs
android.archs = arm64-v8a, armeabi-v7a

# Icons / presplash (add your own artwork here)
# icon.filename = %(source.dir)s/assets/icon.png
# presplash.filename = %(source.dir)s/assets/presplash.png

# Log level (1=error, 2=warning, 3=info, 4=debug)
log_level = 2

# Warn on non-default values
warn_on_root = 1

[buildozer]

# Target platform
android.release = 0
