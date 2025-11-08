import React from "react";
import ReactDOM from "react-dom";
import { StreamlitComponentBase, withStreamlitConnection } from "@streamlit/component-lib";
import { FadeText } from "./FadeText";
import "./index.css";

class FadeTextComponent extends StreamlitComponentBase {
  render() {
    const { text, direction, wordDelay } = this.props.args;

    return (
      <div>
        <FadeText text={text} direction={direction} wordDelay={wordDelay} />
      </div>
    );
  }
}

export default withStreamlitConnection(FadeTextComponent);