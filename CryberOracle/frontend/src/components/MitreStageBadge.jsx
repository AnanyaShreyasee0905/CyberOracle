import React from 'react'

const LABELS = {
  benign: 'Benign',
  reconnaissance: 'Reconnaissance',
  credential_access: 'Credential Access',
  lateral_movement: 'Lateral Movement',
  exfiltration: 'Exfiltration',
}

export default function MitreStageBadge({ stage }) {
  return (
    <span className={`stage-badge stage-${stage}`}>
      {LABELS[stage] || stage}
    </span>
  )
}
