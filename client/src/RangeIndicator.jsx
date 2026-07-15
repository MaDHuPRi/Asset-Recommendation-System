import "./RangeIndicator.css";

/**
 * Signature element of the dashboard: renders the estimated causal uplift
 * as a point on a horizontal scale, with its 90% CI shown as a band and a
 * zero-line marked -- so "is this recommendation actually likely to help"
 * is visible at a glance rather than buried in a number. This is the one
 * place the design takes a real risk; everything else stays quiet.
 */
export default function RangeIndicator({ estimate, ciLow, ciHigh }) {
  const domainMax = Math.max(3, Math.abs(ciLow), Math.abs(ciHigh), Math.abs(estimate)) * 1.15;
  const domainMin = -domainMax;
  const toPct = (v) => ((v - domainMin) / (domainMax - domainMin)) * 100;

  const bandLeft = toPct(ciLow);
  const bandWidth = toPct(ciHigh) - toPct(ciLow);
  const zeroPct = toPct(0);
  const markerPct = toPct(estimate);
  const positive = estimate >= 0;

  return (
    <div className="range-indicator">
      <div className="range-track">
        <div className="range-zero-line" style={{ left: `${zeroPct}%` }} />
        <div
          className="range-band"
          style={{ left: `${bandLeft}%`, width: `${bandWidth}%` }}
          data-positive={positive}
        />
        <div
          className="range-marker"
          style={{ left: `${markerPct}%` }}
          data-positive={positive}
        />
      </div>
      <div className="range-labels">
        <span>{domainMin.toFixed(1)}pp</span>
        <span className="range-labels-zero">0</span>
        <span>{domainMax.toFixed(1)}pp</span>
      </div>
    </div>
  );
}
