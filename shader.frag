#version 330 core
out vec4 FragColor;
in vec2 TexCoord;
uniform int frameCount;
uniform float aspectRatio;
uniform int useFFmpeg;

void main() {
    // Ball properties
    float ballRadius = 0.1;
    vec2 ballPosition;
    
    // Calculate ball position with time
    float speed = 0.01;
    float timeScale = float(frameCount) * speed;
    
    // X position: bounce between 0.1 and 0.9 (accounting for radius)
    float xPeriod = 2.0;
    float xPhase = mod(timeScale, xPeriod);
    float xDirection = xPhase < (xPeriod / 2.0) ? 1.0 : -1.0;
    float xOffset = min(xPhase, xPeriod - xPhase) / (xPeriod / 2.0);
    ballPosition.x = 0.1 + 0.8 * xOffset;
    
    // Y position: bounce between 0.1 and 0.9 (accounting for radius)
    float yPeriod = 1.5;
    float yPhase = mod(timeScale, yPeriod);
    float yOffset = min(yPhase, yPeriod - yPhase) / (yPeriod / 2.0);
    ballPosition.y = 0.1 + 0.8 * yOffset;
    
    // Adjust coordinates for aspect ratio
    vec2 aspectCorrectedCoord = vec2(TexCoord.x, TexCoord.y * aspectRatio);
    vec2 aspectCorrectedBallPos = vec2(ballPosition.x, ballPosition.y * aspectRatio);
    
    // Calculate distance from current fragment to ball center
    float distance = length(aspectCorrectedCoord - aspectCorrectedBallPos);
    
    // Draw the ball and background
    if (distance < ballRadius) {
        // Inside the ball
        vec3 ballColor = vec3(1.0, 0.2, 0.2);
        float shading = 1.0 - distance/ballRadius * 0.5;
        FragColor = vec4(ballColor * shading, 1.0);
    } else {
        // Background with walls (consider aspect ratio for y-axis)
        float wallThickness = 0.02;
        bool isWall = TexCoord.x < wallThickness || TexCoord.x > (1.0 - wallThickness) || 
                      aspectCorrectedCoord.y < wallThickness || aspectCorrectedCoord.y > (1.0 * aspectRatio - wallThickness);
        
        if (isWall) {
            FragColor = vec4(0.5, 0.5, 0.7, 1.0);
        } else {
            // Checkerboard background
            float checkSize = 0.1;
            float check = mod(floor(TexCoord.x / checkSize) + floor(TexCoord.y / checkSize), 2.0);
            vec3 backgroundColor = check > 0.5 ? vec3(0.95) : vec3(0.8);
            FragColor = vec4(backgroundColor, 1.0);
        }
    }
	if (useFFmpeg == 0) {
		vec4 tmpclr = FragColor;
		FragColor = vec4(tmpclr.b, tmpclr.g, tmpclr.r, 1.0);
	}
}
