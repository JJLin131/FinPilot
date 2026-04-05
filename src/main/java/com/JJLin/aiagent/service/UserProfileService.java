package com.JJLin.aiagent.service;

import com.JJLin.aiagent.entites.ChatRequest;
import com.JJLin.aiagent.entites.UserProfile;
import com.JJLin.aiagent.entites.UserProfilePatch;
import com.JJLin.aiagent.extractor.UserProfileExtractorService;
import com.JJLin.aiagent.mapper.UserProfileMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;

@Service
public class UserProfileService {

    private static final Logger log = LoggerFactory.getLogger(UserProfileService.class);

    private final UserProfileMapper userProfileMapper;
    private final UserProfileExtractorService extractorService;

    public UserProfileService(UserProfileMapper userProfileMapper, UserProfileExtractorService extractorService) {
        this.userProfileMapper = userProfileMapper;
        this.extractorService = extractorService;
    }

    public UserProfile getByUserId(String userId) {
        return userId == null || userId.isBlank() ? null : userProfileMapper.selectById(userId);
    }

    @Async("memoryTaskExecutor")
    public void updateFromRequestAsync(ChatRequest request, UserProfile existingProfile) {
        try {
            UserProfilePatch patch = extractorService.extract(request.getContent(), existingProfile);
            if (!patch.isEmpty()) {
                upsert(request.getUserId(), patch);
            }
        } catch (Exception ex) {
            log.warn("user profile async update failed. userId={}", request.getUserId(), ex);
        }
    }

    public void upsert(String userId, UserProfilePatch patch) {
        if (userId == null || userId.isBlank() || patch == null || patch.isEmpty()) {
            return;
        }
        UserProfile profile = userProfileMapper.selectById(userId);
        LocalDateTime now = LocalDateTime.now();
        if (profile == null) {
            profile = new UserProfile();
            profile.setUserId(userId);
            profile.setCreatedAt(now);
        }
        applyPatch(profile, patch);
        profile.setUpdatedAt(now);
        if (profile.getCreatedAt() == null) {
            profile.setCreatedAt(now);
        }
        if (userProfileMapper.selectById(userId) == null) {
            userProfileMapper.insert(profile);
        } else {
            userProfileMapper.updateById(profile);
        }
    }

    private void applyPatch(UserProfile profile, UserProfilePatch patch) {
        if (patch.getAge() != null) {
            profile.setAge(patch.getAge());
        }
        if (!isBlank(patch.getOccupation())) {
            profile.setOccupation(patch.getOccupation());
        }
        if (!isBlank(patch.getEducation())) {
            profile.setEducation(patch.getEducation());
        }
        if (!isBlank(patch.getIncomeRange())) {
            profile.setIncomeRange(patch.getIncomeRange());
        }
        if (!isBlank(patch.getGender())) {
            profile.setGender(patch.getGender());
        }
        if (!isBlank(patch.getCity())) {
            profile.setCity(patch.getCity());
        }
        if (!isBlank(patch.getMaritalStatus())) {
            profile.setMaritalStatus(patch.getMaritalStatus());
        }
        if (!isBlank(patch.getNotes())) {
            profile.setNotes(patch.getNotes());
        }
    }

    private boolean isBlank(String value) {
        return value == null || value.isBlank();
    }
}
