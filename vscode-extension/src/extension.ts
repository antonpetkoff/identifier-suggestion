import axios from 'axios';
import * as vscode from 'vscode';

export function activate(context: vscode.ExtensionContext) {
  context.subscriptions.push(
    vscode.languages.registerCodeActionsProvider(
      'java',
      new MethodNameRecommender(),
      { providedCodeActionKinds: MethodNameRecommender.providedCodeActionKinds }
    )
  );
}

/**
 * Provides code actions for Java method names.
 */
export class MethodNameRecommender implements vscode.CodeActionProvider {

  public static readonly providedCodeActionKinds = [
    vscode.CodeActionKind.QuickFix
  ];

  public async provideCodeActions(
    document: vscode.TextDocument,
    selectedRange: vscode.Range
  ): Promise<vscode.CodeAction[] | undefined> {
    const selectedText = document.getText(selectedRange);

		// TODO: add error handling
    const { data: { predictions } } = await axios.get<{predictions: string[]}>(
      'http://localhost:5000/predict',
      { params: { input: selectedText } }
    );

    const currentMethodNameEnd = selectedText.indexOf('(');

    const currentMethodNameRange = new vscode.Range(
      selectedRange.start,
      selectedRange.start.translate(0, currentMethodNameEnd)
    );

    const fixes = predictions.map(
      suggestion => this.createFix(document, currentMethodNameRange, suggestion)
    )

    // Marking a single fix as `preferred` means that users can apply it with a
    // single keyboard shortcut using the `Auto Fix` command.
    fixes[0].isPreferred = true;

    return fixes;
  }

  private createFix(
    document: vscode.TextDocument,
    range: vscode.Range,
    methodName: string
  ): vscode.CodeAction {
    const fix = new vscode.CodeAction(
      `Replace with ${methodName}`,
      vscode.CodeActionKind.QuickFix
    );

    fix.edit = new vscode.WorkspaceEdit();

    fix.edit.replace(document.uri, range, methodName);

    return fix;
  }
}
