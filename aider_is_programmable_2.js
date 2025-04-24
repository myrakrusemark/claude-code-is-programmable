import { spawn, execSync } from 'child_process';
import { mkdir, existsSync } from 'fs';
import { dirname } from 'path';

// Create directory if it doesn't exist
const todoDir = './cc_todo';
const todoFile = `${todoDir}/todo.ts`;

try {
  // Ensure directory exists
  if (!existsSync(todoDir)) {
    mkdir(todoDir, { recursive: true }, (err) => {
      if (err) throw err;
    });
  }

  // Generate branch name
  const branchName = 'feature-todo-app';

  // 1. Create and checkout a new branch
  console.log(`Creating and checking out new branch: ${branchName}`);
  execSync(`git checkout -b ${branchName}`);

  // 2. Run aider directly with the todo task
  console.log('Running aider to create todo app...');
  const aiderProcess = spawn('aider', [
    '--no-git',
    todoFile,
    '--message',
    'CREATE ./cc_todo/todo.ts: a zero library CLI todo app with basic CRUD.'
  ], {
    stdio: 'inherit'
  });

  // Handle aider completion
  aiderProcess.on('close', (code) => {
    if (code !== 0) {
      console.error(`Aider process exited with code ${code}`);
      cleanupAndExit(1);
      return;
    }

    try {
      // 3. Git operations - stage and commit
      console.log('Staging and committing changes...');
      execSync('git add .');
      execSync('git commit -m "Add TypeScript todo app with CRUD functionality"');

      // 4. Switch back to main branch
      console.log('Switching back to main branch...');
      execSync('git checkout main');

      console.log(`Task completed. Changes committed to branch: ${branchName}`);
    } catch (error) {
      console.error(`Error during git operations: ${error.message}`);
      cleanupAndExit(1);
    }
  });

} catch (error) {
  console.error(`Error: ${error.message}`);
  cleanupAndExit(1);
}

// Helper function to cleanup and exit
function cleanupAndExit(code) {
  try {
    execSync('git checkout main');
  } catch (error) {
    console.error('Failed to switch back to main branch');
  }
  process.exit(code);
}