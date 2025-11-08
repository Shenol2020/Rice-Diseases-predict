import React from "react";
import PropTypes from "prop-types";
import "./FadeText.css";

export function FadeText({ text, direction, wordDelay }) {
  const words = text.split(" ");

  return (
    <div className="fade-text">
      {words.map((word, index) => (
        <span
          key={index}
          className={`fade-${direction}`}
          style={{ animationDelay: `${index * wordDelay}s` }}
        >
          {word}
        </span>
      ))}
    </div>
  );
}

FadeText.propTypes = {
  text: PropTypes.string.isRequired,
  direction: PropTypes.oneOf(["in", "out"]).isRequired,
  wordDelay: PropTypes.number,
};

FadeText.defaultProps = {
  wordDelay: 0.2,
};