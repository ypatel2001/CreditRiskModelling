package handlers

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"os/exec"

	"github.com/gin-gonic/gin"
)

func Predict(c *gin.Context) {
	// Read JSON input
	var profile map[string]interface{}
	if err := c.ShouldBindJSON(&profile); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid input"})
		return
	}

	// Convert profile to JSON
	profileJSON, err := json.Marshal(profile)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to process input"})
		return
	}

	// Call Python predict.py script
	cmd := exec.Command("python3", "../model/predict.py")
	cmd.Stdin = bytes.NewReader(profileJSON)
	var out bytes.Buffer
	cmd.Stdout = &out

	// Capture stderr separately
	var stderr bytes.Buffer
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": fmt.Sprintf("Prediction failed: %v\n%s",
				err, stderr.String()),
		})
		return
	}

	// Parse Python output
	var result map[string]interface{}
	if err := json.Unmarshal([]byte(out.String()), &result); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": fmt.Sprintf("Invalid prediction output format: %v\nstdout: %s",
				err, out.String()),
		})
		return
	}

	// Expect "probability" to be a string now
	probabilityString, ok := result["probability"].(string)
	if !ok {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Invalid probability value"})
		return
	}

	c.JSON(http.StatusOK, gin.H{"probability": probabilityString})
}
