# Java Method Name Recommender

This extension provides Java method name suggestions through code actions
in the editor. Make a selection of the Java method. Then, click on
the Code Action lightbulb or use the Quick Fix command through
the `Ctrl + .` keyboard shortcut. You will see a list of suggested method names
which are sorted by relevancy. Select the on you prefer the most and
the method name will be automatically replaced.

## Demonstration

TODO: add demo gifs

## Configuration

The extension comes with a `methodNameRecommender.serverUrl` configuration parameter
which sets the base URL of the prediction server.
By default, it is `http://localhost:5000`.
You might want to serve predictions somewhere else.

### Building the extension

1. Node.js LTS 12.13.0 or later is required for building the extension.

1. Run `npm install` to install the project dependencies.

1. Run `npm run package` to build and package the extension.

### Installation

1. Visual Studio Code is required, in order to install the extension in it.

1. Run `code --install-extension identifier-recommender-1.0.0.vsix` to install the extension in VSCode.

1. Run the prediction server script in the main project to serve predictions to the extension.

### Uninstallation

1. Run `code --uninstall-extension antonpetkoff.identifier-recommender` to uninstall the extension from VSCode.

You can use `code --list-extensions` to list all installed extensions
or just scroll the extensions inside the editor UI.
