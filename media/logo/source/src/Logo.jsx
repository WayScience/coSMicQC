import { useEffect } from "react";

const colorA = "#581c87";
const colorALight = "#e9d5ff";
const colorB = "#a21caf";
const colorBLight = "#e879f9";
const colorC = "#ec4899";
const colorCLight = "#f472b6";
const colorD = "#0ea5e9";
const darkBg = false;
const text = true;

// const font = "Urbanist";
// const font = "Roboto Condensed";
const font = "Rubik";

const fontUrl = font.replace(" ", "+");

const Logo = () => {
  useEffect(() => {
    fitViewBox();
  });

  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      strokeLinecap="round"
      strokeLinejoin="round"
      style={text ? { fontSize: "35", fontFamily: font } : undefined}
      textAnchor={text ? "middle" : undefined}
      dominantBaseline={text ? "middle" : undefined}
    >
      {text && (
        <style>
          {`@import url("https://fonts.googleapis.com/css2?family=${fontUrl}");`}
        </style>
      )}

      {/* pulsars */}
      <g>
        <path
          d={path(
            wrap(
              repeat(
                [
                  [-10, 40],
                  [4, 60],
                  [10, 40],
                ],
                6
              )
            )
          )}
          fill={colorA}
          transform="rotate(0) scale(1)"
        />
        <path
          d={path(
            wrap(
              repeat(
                [
                  [-10, 35],
                  [6, 60],
                  [10, 35],
                ],
                6
              )
            )
          )}
          fill={colorB}
          transform="rotate(10) scale(0.9)"
        />
        <path
          d={path(
            wrap(
              repeat(
                [
                  [-10, 30],
                  [8, 60],
                  [10, 30],
                ],
                6
              )
            )
          )}
          fill={colorC}
          transform="rotate(20) scale(0.75)"
        />
      </g>

      {/* magnifying glass */}
      <g>
        <circle cx={0} cy={0} r={1.4 * 10} fill={"none"} />
        <path
          d="M -20 20 L -10 10 A 10 10 0 0 0 10 -10 A 10 10 0 0 0 -10 10"
          fill="none"
          stroke="white"
          strokeWidth={5}
        />
      </g>

      {/* stars */}
      <g fill={colorD}>
        <path
          d={path(
            wrap(
              repeat(
                [
                  [0, 7],
                  [5, 0],
                  [10, 7],
                ],
                4
              )
            )
          )}
          transform="translate(38, 38)"
        />
        <path
          d={path(
            wrap(
              repeat(
                [
                  [0, 5],
                  [5, 0],
                  [10, 5],
                ],
                4
              )
            )
          )}
          transform="translate(-40, -35)"
        />
        <path
          d={path(
            wrap(
              repeat(
                [
                  [0, 3],
                  [5, 0],
                  [10, 3],
                ],
                4
              )
            )
          )}
          transform="translate(-30, 40)"
        />
      </g>

      {/* text */}
      {text && (
        <g>
          <text x="75" y="-9" fill={darkBg ? colorALight : colorA}>
            co
          </text>
          <text x="106" y="-9" fill={darkBg ? colorBLight : colorB}>
            S
          </text>
          <text x="129" y="-9" fill={darkBg ? colorBLight : colorB}>
            M
          </text>
          <text x="158" y="-9" fill={darkBg ? colorALight : colorA}>
            ic
          </text>
          <text x="106" y="21" fill={darkBg ? colorCLight : colorC}>
            Q
          </text>
          <text x="129" y="21" fill={darkBg ? colorCLight : colorC}>
            C
          </text>
        </g>
      )}
    </svg>
  );
};

export default Logo;

const repeat = (points, count) => {
  const [min, max] = extent(points);
  points = Array.from({ length: count }, (_, index) =>
    points.map(([x, y]) => [x + (index + 1) * (max - min), y])
  );
  return points;
};

const extent = (points) => [
  Math.min(...points.map(([x]) => x)),
  Math.max(...points.map(([x]) => x)),
];

const wrap = (curves) => {
  const [min, max] = extent(curves.flat());
  curves = curves.map((points) =>
    points.map(([x, y]) => {
      // return [x, y];
      const radius = y;
      const angle = 360 * (x / (max - min));
      return [sin(angle) * radius, cos(angle) * radius];
    })
  );
  return curves;
};

const path = (curves) => {
  const path = [
    ["M", ...curves.at(0).at(0)],
    ...curves.map((points) => ["L", ...points.at(0), "C", ...points.flat()]),
    ["z"],
  ];
  return path
    .map((curve) =>
      curve
        .map((value) =>
          typeof value === "number" ? Math.round(value * 100) / 100 : value
        )
        .join(" ")
    )
    .join(" ");
};

const sin = (degrees) => Math.sin(((2 * Math.PI) / 360) * degrees);
const cos = (degrees) => Math.cos(((2 * Math.PI) / 360) * degrees);

const fitViewBox = () => {
  const padding = 1;
  const svg = document.querySelector("svg");
  let { x, y, width, height } = svg.getBBox();
  x -= padding;
  y -= padding;
  width += 2 * padding;
  height += 2 * padding;
  x -= (round(width) - width) / 2;
  y -= (round(height) - height) / 2;
  width = round(width);
  height = round(height);
  const viewBox = [x, y, width, height].map(Math.round).join(" ");
  svg.setAttribute("viewBox", viewBox);
};

const round = (value, factor = 10) => Math.ceil(value / factor) * factor;
