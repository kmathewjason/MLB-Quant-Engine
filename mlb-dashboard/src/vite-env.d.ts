// Allow CSS module imports
declare module '*.css' {
  const content: Record<string, string>
  export default content
}
