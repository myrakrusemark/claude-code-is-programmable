import { spawn } from 'child_process';

const prompt = `git checkout a NEW branch. 
CREATE ./cc_todo/todo.ts: a zero library CLI todo app with basic CRUD. 
THEN git stage, commit and SWITCH back to main.`;

const command = 'claude';
const args = ['-p', prompt, '--allowedTools', 'Edit', 'Bash', 'Write'];

const child = spawn(command, args, {
  stdio: 'inherit'
});

child.on('close', (code) => {
  console.log(`Claude process exited with code ${code}`);
});