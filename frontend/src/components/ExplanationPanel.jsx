import React from 'react'

export default function ExplanationPanel({ explanations }) {
  return (
    <ul className="explanation-list">
      {explanations.map((text, i) => (
        <li key={i}>{text}</li>
      ))}
    </ul>
  )
}
